#!/usr/bin/env bash

# Train the vertical-gripper ACT pilot with the SAME contract as the dual-camera
# baseline (30k updates, batch size 8, AMP off, seed 1000, all 20 episodes, both
# cameras auto-detected from the dataset). The only difference versus
# act_stack_two_cubes_dualcam_20ep_b8_30k_seed1000 is the training data's gripper
# orientation / wrist viewpoint, so the two checkpoints are directly comparable.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_act_vertical.sh [--dry-run] [RUN_NAME] [STEPS] [BATCH_SIZE] [SAVE_FREQ]

Defaults match the dual-camera baseline contract:
  RUN_NAME   act_stack_two_cubes_vertical_20ep_b8_30k_seed1000
  STEPS      30000
  BATCH_SIZE 8
  SAVE_FREQ  5000
  seed       1000  (override with ACT_VERTICAL_SEED)

The dataset GY-William/lerobot_stack_two_cubes_vertical_20ep must already hold
exactly 20 episodes with both observation.images.wrist and observation.images.front.
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

run_name="${1:-act_stack_two_cubes_vertical_20ep_b8_30k_seed1000}"
steps="${2:-30000}"
batch_size="${3:-8}"
save_freq="${4:-5000}"
seed="${ACT_VERTICAL_SEED:-1000}"

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
  echo "ACT_VERTICAL_SEED must be a non-negative integer." >&2
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
dataset_repo="GY-William/lerobot_stack_two_cubes_vertical_20ep"
dataset_root="${ACT_VERTICAL_DATASET_ROOT:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$dataset_repo}"
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
  printf 'dataset gate:\n  verify %s has 20 episodes + wrist/front cameras\n\n' "$dataset_root"
  printf 'training command:\n  '
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

# Dataset contract gate: exactly 20 episodes and both cameras present. This
# prevents silently training a comparison model on a mis-recorded dataset.
"$python_bin" - "$dataset_root" <<'PY'
import json, sys
info = json.load(open(sys.argv[1] + "/meta/info.json"))
eps = int(info["total_episodes"])
cams = sorted(f for f in info["features"] if f.startswith("observation.images."))
need = {"observation.images.wrist", "observation.images.front"}
problems = []
if eps != 20:
    problems.append(f"expected 20 episodes, found {eps}")
if not need.issubset(set(cams)):
    problems.append(f"missing camera(s); found {cams}")
if problems:
    print("Dataset contract FAILED: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(1)
print(f"Dataset contract OK: {eps} episodes, cameras={cams}")
PY

echo "Starting ACT vertical-gripper run: $run_name"
echo "episodes=20 steps=$steps batch_size=$batch_size AMP=false seed=$seed cameras=wrist+front"
exec "${train_cmd[@]}"
