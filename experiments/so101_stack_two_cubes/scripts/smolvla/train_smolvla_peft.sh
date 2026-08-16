#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_smolvla_peft.sh [--dry-run] RUN_NAME [STEPS] [BATCH_SIZE] [LORA_RANK] [SAVE_FREQ] [MIXED_PRECISION]

Example:
  train_smolvla_peft.sh smolvla_lora_r16_two_orders_20k 20000 1 16 5000 fp16

Defaults: 20,000 steps, batch size 1, LoRA rank 16, save every 5,000
steps, and bf16 mixed precision. Existing output directories are never
overwritten. The merged dataset must pass the two-task balance gate first.
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -lt 1 || $# -gt 6 ]]; then
  usage >&2
  exit 2
fi

run_name="$1"
steps="${2:-20000}"
batch_size="${3:-1}"
lora_rank="${4:-16}"
save_freq="${5:-5000}"
mixed_precision="${6:-bf16}"

if [[ ! "$run_name" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
fi
for value_name in steps batch_size lora_rank save_freq; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name^^} must be a positive integer." >&2
    exit 2
  fi
done
if (( save_freq > steps )); then
  echo "SAVE_FREQ must not exceed STEPS." >&2
  exit 2
fi
if [[ "$mixed_precision" != "fp16" && "$mixed_precision" != "bf16" ]]; then
  echo "MIXED_PRECISION must be fp16 or bf16." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
train_bin="${LEROBOT_TRAIN_BIN:-$(dirname -- "$python_bin")/lerobot-train}"
accelerate_bin="${ACCELERATE_BIN:-$(dirname -- "$python_bin")/accelerate}"
dataset_repo="${LANGUAGE_DATASET_REPO:-GY-William/lerobot_stack_two_orders_language_v2}"
dataset_root="${LANGUAGE_DATASET_ROOT:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$dataset_repo}"
base_model="${SMOLVLA_BASE_MODEL:-/home/hai/models/lerobot_smolvla_base_c83c316}"
backbone_model="${SMOLVLA_BACKBONE_MODEL:-/home/hai/models/smolvlm2_500m_processor_7b375e1}"
base_manifest="${SMOLVLA_BASE_MANIFEST:-$experiment_dir/manifests/smolvla_base.json}"
output_dir="$repo_root/outputs/train/$run_name"
warmup_steps=$(( steps < 500 ? steps / 10 : 500 ))
if (( warmup_steps < 1 )); then warmup_steps=1; fi

# Dataset-gate parameters. The defaults are the two-order language experiment's
# contract; a single-task dataset (e.g. the redleft ACT set reused for a
# SmolVLA-vs-ACT comparison) overrides them via the environment.
min_tasks="${SMOLVLA_MIN_TASKS:-2}"
min_episodes_per_task="${SMOLVLA_MIN_EPISODES_PER_TASK:-30}"
if [[ -n "${SMOLVLA_EXPECTED_TASKS:-}" ]]; then
  expected_tasks="$SMOLVLA_EXPECTED_TASKS"   # newline-separated
else
  expected_tasks=$'Stack the yellow cube on top of the red cube\nStack the red cube on top of the yellow cube'
fi

# Dataloader workers. Defaults to 0 so every checkpoint trained so far stays
# reproducible; video decode is otherwise serialised with the training step and
# becomes the bottleneck on multi-camera datasets. Raise it via the environment
# (about half the cores: 8 on the A4500 host, 4 on the Jetson).
num_workers="${LEROBOT_NUM_WORKERS:-0}"

gate_cmd=(
  "$python_bin" "$script_dir/validate_language_dataset.py" "$dataset_root"
  "--min-tasks=$min_tasks"
  "--min-episodes-per-task=$min_episodes_per_task"
)
while IFS= read -r expected_task; do
  [[ -n "$expected_task" ]] && gate_cmd+=("--expected-task=$expected_task")
done <<< "$expected_tasks"

export PYTHONNOUSERSITE=1

# Fine-tuning method.
#
#   expert_only (default) -- what the released checkpoint's own config asks for:
#     freeze_vision_encoder and train_expert_only are already true in it, so the
#     VLM is frozen and only the action expert trains. That is parameter
#     efficient on its own; stacking LoRA on top is redundant. Optimizer and
#     scheduler settings are inherited from the checkpoint rather than
#     overridden, which is the point of following the published recipe.
#
#   lora -- adds a LoRA adapter and the higher learning rate that needs. Kept
#     for the two-order language experiment, which was designed around it.
#
# Anything trained one way is not comparable with the other; say which was used.
finetune_mode="${SMOLVLA_FINETUNE_MODE:-expert_only}"
case "$finetune_mode" in
  expert_only) method_args=() ;;
  lora)
    method_args=(
      --policy.optimizer_lr=1e-3
      --policy.scheduler_decay_lr=1e-4
      "--policy.scheduler_warmup_steps=$warmup_steps"
      "--policy.scheduler_decay_steps=$steps"
      --peft.method_type=LORA
      "--peft.r=$lora_rank"
    )
    ;;
  *)
    echo "SMOLVLA_FINETUNE_MODE must be 'expert_only' or 'lora', got '$finetune_mode'." >&2
    exit 2
    ;;
esac

train_cmd=(
  "$accelerate_bin" launch
  "--mixed_precision=$mixed_precision"
  --num_processes=1
  "$train_bin"
  "--policy.path=$base_model"
  --policy.input_features=null
  --policy.output_features=null
  --policy.device=cuda
  --policy.push_to_hub=false
  "--policy.vlm_model_name=$backbone_model"
  --policy.load_vlm_weights=false
  "${method_args[@]}"
  "--dataset.repo_id=$dataset_repo"
  "--dataset.root=$dataset_root"
  "--output_dir=$output_dir"
  "--job_name=$run_name"
  "--batch_size=$batch_size"
  "--num_workers=$num_workers"
  "--steps=$steps"
  --log_freq=50
  --save_checkpoint=true
  "--save_freq=$save_freq"
  --seed=1000
  --wandb.enable=false
)

if "$dry_run"; then
  printf 'base-model gate:\n  '
  printf '%q ' \
    "$python_bin" "$script_dir/verify_smolvla_base.py" \
    "--root=$base_model" "--backbone-root=$backbone_model" "--manifest=$base_manifest"
  printf '\n\ndataset gate:\n  '
  printf '%q ' "${gate_cmd[@]}"
  printf '\n\ntraining command:\n  '
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in \
  "$python_bin" \
  "$train_bin" \
  "$accelerate_bin" \
  "$base_manifest" \
  "$base_model/config.json" \
  "$base_model/model.safetensors" \
  "$backbone_model/config.json" \
  "$backbone_model/tokenizer.json" \
  "$backbone_model/preprocessor_config.json" \
  "$dataset_root/meta/info.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done
"$python_bin" "$script_dir/verify_smolvla_base.py" \
  "--root=$base_model" \
  "--backbone-root=$backbone_model" \
  "--manifest=$base_manifest"
if ! "$python_bin" -c 'import num2words, peft' >/dev/null 2>&1; then
  echo 'Missing SmolVLA/PEFT dependencies. Install the validated versions with:' >&2
  echo '  python -m pip install --index-url https://pypi.org/simple num2words==0.5.14 peft==0.20.0' >&2
  exit 1
fi
if ! "$python_bin" -c 'import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)' \
  >/dev/null 2>&1 && [[ "$mixed_precision" == "bf16" ]]; then
  echo "BF16 is not supported on this CUDA device; rerun with MIXED_PRECISION=fp16." >&2
  exit 1
fi
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output directory: $output_dir" >&2
  exit 1
fi

"${gate_cmd[@]}"

echo "Starting SmolVLA LoRA training: $run_name"
echo "steps=$steps batch_size=$batch_size rank=$lora_rank mixed_precision=$mixed_precision"
exec "${train_cmd[@]}"
