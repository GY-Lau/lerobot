#!/usr/bin/env bash

# Train an ACT model on any local dataset with the fixed experiment contract
# (30k updates, batch 8, AMP off, seed 1000, both cameras auto-detected). Used
# for the red-left placement study: one model on the 20 red-left episodes, one on
# the combined 40 (previous vertical 20 + red-left 20). Only the dataset changes,
# so the resulting checkpoints are comparable to each other and to the earlier runs.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_act_generic.sh [--dry-run] DATASET_REPO_ID DATASET_ROOT RUN_NAME [EXPECTED_EPISODES]

Defaults: STEPS 30000, BATCH 8, SAVE_FREQ 5000, seed 1000 (override with ACT_SEED).
If EXPECTED_EPISODES is given, the dataset must contain exactly that many episodes.
The dataset must have observation.images.wrist (and, if present, front is used too).
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

dataset_repo="$1"
dataset_root="$2"
run_name="$3"
expected_episodes="${4:-}"
steps="${ACT_STEPS:-30000}"
batch_size="${ACT_BATCH:-8}"
save_freq="${ACT_SAVE_FREQ:-5000}"
seed="${ACT_SEED:-1000}"

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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
train_bin="${LEROBOT_TRAIN_BIN:-$(dirname -- "$python_bin")/lerobot-train}"
output_dir="$repo_root/outputs/train/$run_name"

export PYTHONNOUSERSITE=1

# Dataloader workers. Defaults to 0 so every checkpoint trained so far stays
# reproducible; video decode is otherwise serialised with the training step and
# becomes the bottleneck on multi-camera datasets. Raise it via the environment
# (about half the cores: 8 on the A4500 host, 4 on the Jetson).
num_workers="${LEROBOT_NUM_WORKERS:-0}"

train_cmd=(
  "$train_bin"
  "--dataset.repo_id=$dataset_repo"
  "--dataset.root=$dataset_root"
  --policy.type=act
  --policy.device=cuda
  --policy.use_amp=false
  --policy.push_to_hub=false
  "--output_dir=$output_dir"
  "--job_name=$run_name"
  "--batch_size=$batch_size"
  "--num_workers=$num_workers"
  "--steps=$steps"
  --log_freq=100
  --save_checkpoint=true
  "--save_freq=$save_freq"
  "--seed=$seed"
  --wandb.enable=false
)

if "$dry_run"; then
  printf 'dataset: %s (%s)\nexpected_episodes: %s\n\ntraining command:\n  ' \
    "$dataset_repo" "$dataset_root" "${expected_episodes:-any}"
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$python_bin" "$train_bin" "$dataset_root/meta/info.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output directory: $output_dir" >&2
  exit 1
fi

"$python_bin" - "$dataset_root" "${expected_episodes:-0}" <<'PY'
import json, sys
info = json.load(open(sys.argv[1] + "/meta/info.json"))
eps = int(info["total_episodes"])
cams = sorted(f for f in info["features"] if f.startswith("observation.images."))
expected = int(sys.argv[2])
problems = []
if expected and eps != expected:
    problems.append(f"expected {expected} episodes, found {eps}")
if "observation.images.wrist" not in cams:
    problems.append(f"missing wrist camera; found {cams}")
if problems:
    print("Dataset contract FAILED: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(1)
print(f"Dataset contract OK: {eps} episodes, cameras={cams}")
PY

echo "Starting ACT run: $run_name"
echo "dataset=$dataset_repo steps=$steps batch_size=$batch_size AMP=false seed=$seed"
exec "${train_cmd[@]}"
