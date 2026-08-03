#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_diffusion.sh [--dry-run] RUN_NAME [STEPS] [BATCH_SIZE] [USE_AMP] [SAVE_FREQ]

Example:
  train_diffusion.sh diffusion_stack_two_cubes_30k 30000 2 true 5000

The defaults are 30,000 steps, batch size 2, AMP enabled, and a checkpoint
every 5,000 steps. RUN_NAME is also used as the output directory name below
outputs/train/. Existing output directories are never overwritten.
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi

if [[ $# -lt 1 || $# -gt 5 ]]; then
  usage >&2
  exit 2
fi

run_name="$1"
steps="${2:-30000}"
batch_size="${3:-2}"
use_amp="${4:-true}"
save_freq="${5:-5000}"

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
if (( save_freq > steps )); then
  echo "SAVE_FREQ must not exceed STEPS." >&2
  exit 2
fi
if [[ "$use_amp" != "true" && "$use_amp" != "false" ]]; then
  echo "USE_AMP must be true or false." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
train_bin="${LEROBOT_TRAIN_BIN:-/home/hai/miniconda3/envs/lerobot/bin/lerobot-train}"
dataset_root="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/GY-William/lerobot_stack_two_cubes"
output_dir="$repo_root/outputs/train/$run_name"

export PYTHONNOUSERSITE=1

train_cmd=(
  "$train_bin"
  --dataset.repo_id=GY-William/lerobot_stack_two_cubes
  "--dataset.root=$dataset_root"
  --policy.type=diffusion
  --policy.device=cuda
  "--policy.use_amp=$use_amp"
  --policy.push_to_hub=false
  "--output_dir=$output_dir"
  "--job_name=$run_name"
  "--batch_size=$batch_size"
  --num_workers=0
  "--steps=$steps"
  --log_freq=100
  --save_checkpoint=true
  "--save_freq=$save_freq"
  --seed=1000
  --wandb.enable=false
)

if $dry_run; then
  printf 'training command:\n  '
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$train_bin" "$dataset_root/meta/info.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output directory: $output_dir" >&2
  exit 1
fi

echo "Starting Diffusion Policy training: $run_name"
echo "steps=$steps batch_size=$batch_size AMP=$use_amp save_freq=$save_freq"
exec "${train_cmd[@]}"
