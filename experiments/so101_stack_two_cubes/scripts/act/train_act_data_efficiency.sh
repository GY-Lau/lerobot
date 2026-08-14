#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_act_data_efficiency.sh [--dry-run] EPISODE_COUNT RUN_NAME [STEPS] [BATCH_SIZE] [SAVE_FREQ]

Example:
  train_act_data_efficiency.sh 10 act_stack_two_cubes_10ep_30k 30000 2 5000

EPISODE_COUNT must exist in act_data_subsets.json (10, 20, or 30). Training
matches the original ACT baseline: 30,000 updates, batch size 2, AMP disabled,
seed 1000, and unchanged policy defaults. Existing outputs are never replaced.
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

if [[ ! "$episode_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPISODE_COUNT must be a positive integer." >&2
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
if (( save_freq > steps )); then
  echo "SAVE_FREQ must not exceed STEPS." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
train_bin="${LEROBOT_TRAIN_BIN:-$(dirname -- "$python_bin")/lerobot-train}"
dataset_repo="GY-William/lerobot_stack_two_cubes"
dataset_root="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$dataset_repo"
positions_csv="$experiment_dir/manifests/training_start_positions.csv"
manifest="$experiment_dir/manifests/act_data_subsets.json"
output_dir="$repo_root/outputs/train/$run_name"
manifest_python="$python_bin"
if [[ ! -x "$manifest_python" ]]; then
  manifest_python="$(command -v python3)"
fi

if ! episodes_json="$($manifest_python -c \
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
  --seed=1000
  --wandb.enable=false
)

if "$dry_run"; then
  printf 'manifest gate:\n  %q %q %q %q %q %q\n\ntraining command:\n  ' \
    "$python_bin" "$script_dir/generate_act_subsets.py" "$positions_csv" --output "$manifest" --check
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

"$python_bin" "$script_dir/generate_act_subsets.py" "$positions_csv" --output "$manifest" --check
echo "Starting matched ACT data-efficiency run: $run_name"
echo "episodes=$episode_count steps=$steps batch_size=$batch_size AMP=false seed=1000"
exec "${train_cmd[@]}"
