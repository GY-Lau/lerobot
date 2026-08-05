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
experiment_dir="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
train_bin="${LEROBOT_TRAIN_BIN:-$(dirname -- "$python_bin")/lerobot-train}"
accelerate_bin="${ACCELERATE_BIN:-$(dirname -- "$python_bin")/accelerate}"
dataset_repo="${LANGUAGE_DATASET_REPO:-GY-William/lerobot_stack_two_orders_language}"
dataset_root="${LANGUAGE_DATASET_ROOT:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$dataset_repo}"
base_model="${SMOLVLA_BASE_MODEL:-lerobot/smolvla_base}"
output_dir="$repo_root/outputs/train/$run_name"
warmup_steps=$(( steps < 500 ? steps / 10 : 500 ))
if (( warmup_steps < 1 )); then warmup_steps=1; fi

export PYTHONNOUSERSITE=1

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
  --policy.load_vlm_weights=true
  --policy.optimizer_lr=1e-3
  --policy.scheduler_decay_lr=1e-4
  "--policy.scheduler_warmup_steps=$warmup_steps"
  "--policy.scheduler_decay_steps=$steps"
  "--dataset.repo_id=$dataset_repo"
  "--dataset.root=$dataset_root"
  --peft.method_type=LORA
  "--peft.r=$lora_rank"
  "--output_dir=$output_dir"
  "--job_name=$run_name"
  "--batch_size=$batch_size"
  --num_workers=0
  "--steps=$steps"
  --log_freq=50
  --save_checkpoint=true
  "--save_freq=$save_freq"
  --seed=1000
  --wandb.enable=false
)

if "$dry_run"; then
  printf 'dataset gate:\n  %q %q %q %q %q\n\ntraining command:\n  ' \
    "$python_bin" "$script_dir/validate_language_dataset.py" "$dataset_root" \
    --min-tasks=2 --min-episodes-per-task=30
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$python_bin" "$train_bin" "$accelerate_bin" "$dataset_root/meta/info.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done
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

"$python_bin" "$script_dir/validate_language_dataset.py" "$dataset_root" \
  --min-tasks=2 \
  --min-episodes-per-task=30 \
  --expected-task='Stack the yellow cube on top of the red cube' \
  --expected-task='Stack the red cube on top of the yellow cube'

echo "Starting SmolVLA LoRA training: $run_name"
echo "steps=$steps batch_size=$batch_size rank=$lora_rank mixed_precision=$mixed_precision"
exec "${train_cmd[@]}"
