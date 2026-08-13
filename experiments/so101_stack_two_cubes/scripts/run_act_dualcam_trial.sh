#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_act_dualcam_trial.sh [--dry-run] RUN_ID [N_ACTION_STEPS]

Example:
  run_act_dualcam_trial.sh eval_act_dualcam_n20_protocol_v1 20

Each invocation records one 20-second evaluation episode. Reuse the same
RUN_ID to append another episode after resetting the cubes and follower.
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

run_id="$1"
n_action_steps="${2:-20}"

if [[ ! "$run_id" =~ ^eval_[a-zA-Z0-9._-]+$ ]]; then
  echo "RUN_ID must begin with eval_ and contain only letters, numbers, dot, underscore, or hyphen." >&2
  exit 2
fi
if [[ ! "$n_action_steps" =~ ^[0-9]+$ ]] || (( n_action_steps < 1 || n_action_steps > 100 )); then
  echo "N_ACTION_STEPS must be an integer from 1 to 100." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
record_bin="$(dirname -- "$python_bin")/lerobot-record"
train_run="act_stack_two_cubes_dualcam_20ep_b8_30k_seed1000"
model="$repo_root/outputs/train/$train_run/checkpoints/030000/pretrained_model"
dataset_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
dataset_root="$dataset_base/GY-William/$run_id"
latency_log="$repo_root/outputs/eval_latency/$run_id.csv"

export PYTHONNOUSERSITE=1

pose_cmd=(
  "$python_bin"
  "$script_dir/check_start_pose.py"
  --profile dualcam20
)

record_cmd=(
  "$record_bin"
  --robot.type=so101_follower
  --robot.port=/dev/ttyACM0
  --robot.id=lerobot_follower_arm
  '--robot.cameras={"front":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'
  --policy.type=act
  "--policy.pretrained_path=$model"
  --policy.device=cuda
  --policy.use_amp=false
  "--policy.n_action_steps=$n_action_steps"
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
  --play_sounds=false
  "--latency_log_path=$latency_log"
  --latency_warmup_frames=30
)

if [[ -f "$dataset_root/meta/info.json" ]]; then
  if [[ ! -f "$dataset_root/meta/tasks.parquet" ]]; then
    echo "Existing evaluation dataset is incomplete: $dataset_root" >&2
    echo "Use a new RUN_ID; this script will not delete the incomplete directory." >&2
    exit 1
  fi
  record_cmd+=(--resume=true)
elif [[ -e "$dataset_root" ]]; then
  echo "Dataset path exists but has no complete metadata: $dataset_root" >&2
  echo "Use a new RUN_ID; this script will not delete the existing path." >&2
  exit 1
fi

if "$dry_run"; then
  printf 'pose check:\n  '
  printf '%q ' "${pose_cmd[@]}"
  printf '\n\nrecord command:\n  '
  printf '%q ' "${record_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in \
  "$python_bin" \
  "$record_bin" \
  "$model/model.safetensors" \
  /dev/ttyACM0 \
  /dev/video0 \
  /dev/video2; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

echo "Checking the dual-camera follower start pose..."
"${pose_cmd[@]}"

echo
echo "Starting one dual-camera ACT trial"
echo "  model: $train_run/030000"
echo "  cameras: front=/dev/video2, wrist=/dev/video0"
echo "  n_action_steps: $n_action_steps"
echo "  dataset: GY-William/$run_id"
echo "Keep hands clear and be ready to stop the robot."
"${record_cmd[@]}"
