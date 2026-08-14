#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_act_v2.sh [--dry-run] EPISODE_COUNT RUN_NAME [STEPS] [BATCH_SIZE] [SAVE_FREQ]

EPISODE_COUNT must be 30 or 50 and must exist in act_v2_subsets.json.
Defaults preserve the v1 comparison contract: 30,000 updates, batch size 2,
save every 5,000 steps, AMP disabled, seed 1000. Set ACT_V2_SEED to run a
different, explicit random seed without changing the default comparison.
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -lt 2 || $# -gt 5 ]]; then
  usage >&2
  exit 2
fi

episode_count="$1"
run_name="$2"
steps="${3:-30000}"
batch_size="${4:-2}"
save_freq="${5:-5000}"
seed="${ACT_V2_SEED:-1000}"

if [[ "$episode_count" != 30 && "$episode_count" != 50 ]]; then
  echo "EPISODE_COUNT must be 30 or 50." >&2
  exit 2
fi
if [[ ! "$run_name" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
fi
for value_name in steps batch_size save_freq; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name^^} must be a positive integer." >&2
    exit 2
  fi
done
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "ACT_V2_SEED must be a non-negative integer." >&2
  exit 2
fi
if (( save_freq > steps )); then
  echo "SAVE_FREQ must not exceed STEPS." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
train_bin="${LEROBOT_TRAIN_BIN:-$(dirname -- "$python_bin")/lerobot-train}"
dataset_repo="GY-William/lerobot_stack_two_cubes_v2"
dataset_root="${ACT_V2_DATASET_ROOT:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$dataset_repo}"
positions_csv="${ACT_V2_POSITIONS_CSV:-$experiment_dir/manifests/act_v2_start_positions.csv}"
manifest="${ACT_V2_MANIFEST:-$experiment_dir/manifests/act_v2_subsets.json}"
output_dir="$repo_root/outputs/train/$run_name"

if [[ ! -f "$manifest" ]]; then
  echo "Missing $manifest; run prepare_act_v2_subsets.sh after recording all 50 episodes." >&2
  exit 1
fi
if ! episodes_json="$($python_bin -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d["subsets"][sys.argv[2]]["episodes"]))' \
  "$manifest" "$episode_count" 2>/dev/null)"; then
  echo "EPISODE_COUNT $episode_count is absent from $manifest" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1

train_cmd=(
  "$train_bin"
  "--dataset.repo_id=$dataset_repo"
  "--dataset.root=$dataset_root"
  "--dataset.episodes=$episodes_json"
  --policy.type=act
  --policy.device=cuda
  --policy.use_amp=false
  --policy.push_to_hub=false
  "--output_dir=$output_dir"
  "--job_name=$run_name"
  "--batch_size=$batch_size"
  --num_workers=0
  "--steps=$steps"
  --log_freq=100
  --save_checkpoint=true
  "--save_freq=$save_freq"
  "--seed=$seed"
  --wandb.enable=false
)

# Optional, explicit augmentation hook used by the robustness experiment. The
# baseline path leaves this unset, so its serialized training contract remains
# unchanged. Keep the full transform dictionary in one JSON value: partial map
# overrides are easy to mis-serialize through draccus.
if [[ -n "${ACT_V2_IMAGE_TRANSFORMS_JSON:-}" ]]; then
  max_num_transforms="${ACT_V2_MAX_NUM_TRANSFORMS:-2}"
  if [[ ! "$max_num_transforms" =~ ^[1-9][0-9]*$ ]]; then
    echo "ACT_V2_MAX_NUM_TRANSFORMS must be a positive integer." >&2
    exit 2
  fi
  train_cmd+=(
    --dataset.image_transforms.enable=true
    "--dataset.image_transforms.max_num_transforms=$max_num_transforms"
    --dataset.image_transforms.random_order=false
    "--dataset.image_transforms.tfs=$ACT_V2_IMAGE_TRANSFORMS_JSON"
  )
fi

if "$dry_run"; then
  printf 'quality gate:\n  %q %q %q %q\n\nmanifest gate:\n  ' \
    "$python_bin" "$script_dir/../common/validate_act_v2_positions.py" "$positions_csv" --expected-episodes=50
  printf '%q ' "$python_bin" "$script_dir/generate_act_subsets.py" "$positions_csv" \
    --sizes 30 50 "--dataset-repo-id=$dataset_repo" "--output=$manifest" --check
  printf '\n\ntraining command:\n  '
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$python_bin" "$train_bin" "$dataset_root/meta/info.json" "$positions_csv" "$manifest"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output directory: $output_dir" >&2
  exit 1
fi

"$python_bin" "$script_dir/../common/validate_act_v2_positions.py" "$positions_csv" --expected-episodes=50
"$python_bin" "$script_dir/generate_act_subsets.py" "$positions_csv" \
  --sizes 30 50 "--dataset-repo-id=$dataset_repo" "--output=$manifest" --check
echo "Starting ACT v2 run: $run_name"
echo "episodes=$episode_count steps=$steps batch_size=$batch_size AMP=false seed=$seed"
exec "${train_cmd[@]}"
