#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_smolvla_language_trial.sh [--dry-run] CONDITION [TRAIN_RUN] [CHECKPOINT]

CONDITION must be one of:
  yellow_exact       Stack the yellow cube on top of the red cube
  yellow_paraphrase  Put the yellow block on the red block
  red_exact          Stack the red cube on top of the yellow cube
  red_paraphrase     Put the red block on the yellow block

Defaults:
  TRAIN_RUN=smolvla_lora_r16_two_orders_20k
  CHECKPOINT=020000

Each invocation records one 20-second evaluation episode. Reusing a condition
appends to its dataset, allowing 10 matched trials per condition.
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

condition="$1"
train_run="${2:-smolvla_lora_r16_two_orders_20k}"
checkpoint="${3:-020000}"

case "$condition" in
  yellow_exact)
    prompt='Stack the yellow cube on top of the red cube'
    ;;
  yellow_paraphrase)
    prompt='Put the yellow block on the red block'
    ;;
  red_exact)
    prompt='Stack the red cube on top of the yellow cube'
    ;;
  red_paraphrase)
    prompt='Put the red block on the yellow block'
    ;;
  *)
    echo "Unknown CONDITION: $condition" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! "$train_run" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "TRAIN_RUN may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
fi
if [[ ! "$checkpoint" =~ ^[0-9]{6}$ ]]; then
  echo "CHECKPOINT must be six digits, for example 020000." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
record_bin="$(dirname -- "$python_bin")/lerobot-record"
model="$repo_root/outputs/train/$train_run/checkpoints/$checkpoint/pretrained_model"
run_id="eval_smolvla_$condition"
dataset_repo="GY-William/$run_id"
dataset_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
dataset_root="$dataset_base/$dataset_repo"
latency_log="$repo_root/outputs/eval_latency/$run_id.csv"

export PYTHONNOUSERSITE=1

record_cmd=(
  "$record_bin"
  --robot.type=so101_follower
  --robot.port=/dev/ttyACM0
  --robot.id=lerobot_follower_arm
  '--robot.cameras={"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'
  "--policy.path=$model"
  --policy.device=cuda
  --policy.use_amp=true
  --policy.push_to_hub=false
  "--dataset.repo_id=$dataset_repo"
  "--dataset.root=$dataset_root"
  "--dataset.single_task=$prompt"
  --dataset.num_episodes=1
  --dataset.episode_time_s=20
  --dataset.reset_time_s=0
  --dataset.fps=30
  --dataset.push_to_hub=false
  --display_data=false
  "--latency_log_path=$latency_log"
  --latency_warmup_frames=30
)

if [[ -f "$dataset_root/meta/info.json" ]]; then
  if ! "$dry_run"; then
    recorded_episodes="$("$python_bin" -c '
import json
import sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["total_episodes"]))
' "$dataset_root/meta/info.json")"
    if (( recorded_episodes >= 10 )); then
      echo "Condition $condition already has $recorded_episodes/10 episodes; refusing trial 11." >&2
      exit 1
    fi
  fi
  record_cmd+=(--resume=true)
fi

if "$dry_run"; then
  printf 'condition: %s\nprompt: %s\n\npose check:\n  ' "$condition" "$prompt"
  printf '%q ' "$python_bin" "$script_dir/check_start_pose.py" --profile relaxed
  printf '\n\nrecord command:\n  '
  printf '%q ' "${record_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in \
  "$python_bin" \
  "$record_bin" \
  "$model/adapter_config.json" \
  "$model/adapter_model.safetensors" \
  /dev/ttyACM0 \
  /dev/video0; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

echo "Checking the follower start pose..."
"$python_bin" "$script_dir/check_start_pose.py" --profile relaxed

echo
echo "Starting one SmolVLA language trial: $condition"
echo "Prompt: $prompt"
echo "Keep hands clear and be ready to stop the robot."
"${record_cmd[@]}"
