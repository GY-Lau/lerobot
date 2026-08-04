#!/usr/bin/env bash

set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: record_inverse_language_data.sh [--dry-run]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
record_bin="$(dirname -- "$python_bin")/lerobot-record"
dataset_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
repo_id="GY-William/lerobot_stack_red_on_yellow"
dataset_root="$dataset_base/$repo_id"

export PYTHONNOUSERSITE=1

record_cmd=(
  "$record_bin"
  --robot.type=so101_follower
  --robot.port=/dev/ttyACM0
  --robot.id=lerobot_follower_arm
  '--robot.cameras={"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'
  --teleop.type=so101_leader
  --teleop.port=/dev/ttyACM1
  --teleop.id=lerobot_leader_arm
  "--dataset.repo_id=$repo_id"
  "--dataset.root=$dataset_root"
  '--dataset.single_task=Stack the red cube on top of the yellow cube'
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
  printf 'pose check:\n  %q %q %q %q\n\nrecord command:\n  ' \
    "$python_bin" "$script_dir/check_start_pose.py" --profile relaxed
  printf '%q ' "${record_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$python_bin" "$record_bin" /dev/ttyACM0 /dev/ttyACM1 /dev/video0; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

"$python_bin" "$script_dir/check_start_pose.py" --profile relaxed
echo "Recording one 20-second demonstration: red cube on yellow cube."
echo "Repeat this command until the dataset contains 30 successful episodes."
exec "${record_cmd[@]}"
