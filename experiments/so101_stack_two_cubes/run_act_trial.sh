#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_act_trial.sh [--dry-run] RUN_ID [CHECKPOINT] [USE_AMP]

Examples:
  run_act_trial.sh act_30k_fixed_20s 030000 false
  run_act_trial.sh act_60k_fixed_20s 060000 true

Each invocation records exactly one 20-second episode. Reusing RUN_ID appends
one episode to the same local evaluation dataset.
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

run_id="$1"
checkpoint="${2:-030000}"
use_amp="${3:-false}"

if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
fi
if [[ ! "$checkpoint" =~ ^[0-9]{6}$ ]]; then
  echo "CHECKPOINT must be six digits, for example 030000." >&2
  exit 2
fi
if [[ "$use_amp" != "true" && "$use_amp" != "false" ]]; then
  echo "USE_AMP must be true or false." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
record_bin="$(dirname -- "$python_bin")/lerobot-record"
model="$repo_root/outputs/train/act_stack_two_cubes_30k/checkpoints/$checkpoint/pretrained_model"
dataset_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
dataset_root="$dataset_base/GY-William/$run_id"

export PYTHONNOUSERSITE=1

record_cmd=(
  "$record_bin"
  --robot.type=so101_follower
  --robot.port=/dev/ttyACM0
  --robot.id=lerobot_follower_arm
  '--robot.cameras={"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'
  --policy.type=act
  "--policy.pretrained_path=$model"
  --policy.device=cuda
  "--policy.use_amp=$use_amp"
  --policy.push_to_hub=false
  "--dataset.repo_id=GY-William/$run_id"
  "--dataset.root=$dataset_root"
  '--dataset.single_task=Stack the yellow cube on top of the red cube'
  --dataset.num_episodes=1
  --dataset.episode_time_s=20
  --dataset.reset_time_s=0
  --dataset.fps=30
  --dataset.push_to_hub=false
  --display_data=false
)

if [[ -f "$dataset_root/meta/info.json" ]]; then
  record_cmd+=(--resume=true)
fi

if "$dry_run"; then
  printf 'pose check:\n  %q %q\n\nrecord command:\n  ' "$python_bin" "$script_dir/check_start_pose.py"
  printf '%q ' "${record_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$python_bin" "$record_bin" "$model/model.safetensors" /dev/ttyACM0 /dev/video0; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

echo "Checking the follower start pose..."
"$python_bin" "$script_dir/check_start_pose.py"

echo
echo "Starting one ACT evaluation trial: $run_id ($checkpoint, AMP=$use_amp)"
echo "Keep hands clear and be ready to stop the robot."
"${record_cmd[@]}"

