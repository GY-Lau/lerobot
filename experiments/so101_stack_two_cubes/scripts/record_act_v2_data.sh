#!/usr/bin/env bash

set -euo pipefail

dry_run=false
status_only=false
case "${1:-}" in
  --dry-run)
    dry_run=true
    shift
    ;;
  --status)
    status_only=true
    shift
    ;;
esac
if [[ $# -ne 0 ]]; then
  echo "Usage: record_act_v2_data.sh [--dry-run|--status]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
record_bin="$(dirname -- "$python_bin")/lerobot-record"
dataset_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
repo_id="GY-William/lerobot_stack_two_cubes_v2"
dataset_root="$dataset_base/$repo_id"
target_episodes=50

export PYTHONNOUSERSITE=1

episode_count="$($python_bin - "$dataset_root/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
else:
    print(int(json.loads(path.read_text(encoding="utf-8"))["total_episodes"]))
PY
)"
if [[ ! "$episode_count" =~ ^[0-9]+$ ]]; then
  echo "Could not determine the ACT v2 dataset episode count." >&2
  exit 1
fi

if [[ -d "$dataset_root" && "$episode_count" == 0 ]]; then
  if "$status_only"; then
    echo "ACT v2 dataset: 0/$target_episodes recorded episodes (incomplete empty directory detected)"
    echo "root: $dataset_root"
    echo "The next recording attempt will preserve this directory as a backup and create a clean dataset."
    exit 0
  fi
  incomplete_backup="${dataset_root}_incomplete_$(date +%Y%m%d_%H%M%S)"
  if "$dry_run"; then
    echo "Would preserve incomplete zero-episode dataset at: $incomplete_backup"
  else
    mv -- "$dataset_root" "$incomplete_backup"
    echo "Preserved incomplete zero-episode dataset at: $incomplete_backup"
  fi
fi

if (( episode_count > 0 )); then
  if [[ ! -f "$dataset_root/meta/tasks.parquet" ]] \
    || ! find "$dataset_root/meta/episodes" -type f -name '*.parquet' -print -quit | grep -q . \
    || ! find "$dataset_root/data" -type f -name '*.parquet' -print -quit | grep -q .; then
    echo "ACT v2 metadata is incomplete despite declaring $episode_count episodes; refusing to resume." >&2
    echo "Review and repair or explicitly archive: $dataset_root" >&2
    exit 1
  fi
fi

if "$status_only"; then
  echo "ACT v2 dataset: $episode_count/$target_episodes recorded episodes"
  echo "root: $dataset_root"
  exit 0
fi

if (( episode_count >= target_episodes )); then
  echo "ACT v2 already contains $episode_count episodes; refusing to exceed the $target_episodes-episode protocol." >&2
  exit 1
fi

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
  '--dataset.single_task=Stack the yellow cube on top of the red cube'
  --dataset.num_episodes=1
  --dataset.episode_time_s=20
  --dataset.reset_time_s=0
  --dataset.fps=30
  --dataset.push_to_hub=false
  --display_data=false
)

if (( episode_count > 0 )); then
  record_cmd+=(--resume=true)
fi

if "$dry_run"; then
  printf 'progress: %d/%d recorded episodes; next episode: %d\n\n' \
    "$episode_count" "$target_episodes" "$(( episode_count + 1 ))"
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
echo "ACT v2 episode $(( episode_count + 1 ))/$target_episodes: yellow cube on red cube."
echo "Keep both cubes visible; use a clean single attempt; lower and center before release."
echo "Loading LeRobot modules on Jetson takes about 25-30 seconds; recording starts after 'Recording episode'."
exec "${record_cmd[@]}"
