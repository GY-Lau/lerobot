#!/usr/bin/env bash

# Record one physical evaluation episode for a vertical-gripper ACT checkpoint.
# Two variants share this runner so the only eval difference is the camera set:
#   wristfront  -> wrist + front cameras, act_..._vertical_20ep_b8_30k_seed1000
#   wristonly   -> wrist camera only,     act_..._vertical_20ep_wristonly_b8_30k_seed1000
#   redleft     -> wrist + front cameras, act_..._redleft_20ep_b8_30k_seed1000
#   combined    -> wrist + front cameras, act_..._combined_40ep_b8_30k_seed1000
# Everything else (vertical start-pose gate, 20 s horizon, 30 FPS, latency log)
# is held fixed, so wristfront vs wristonly is a clean camera ablation and
# redleft vs combined is a clean data ablation for placement precision.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_act_vertical_trial.sh [--dry-run] {wristfront|wristonly|redleft|combined} RUN_ID [N_ACTION_STEPS]

Examples:
  run_act_vertical_trial.sh wristfront eval_act_vertical_wf_n20_v1 20
  run_act_vertical_trial.sh wristonly  eval_act_vertical_wo_n20_v1 20
  run_act_vertical_trial.sh redleft    eval_act_redleft_n100_v1   100
  run_act_vertical_trial.sh combined   eval_act_combined_n100_v1  100

Each invocation records one 20-second evaluation episode. Reuse the same
RUN_ID to append another episode after resetting the cubes and follower.
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

variant="$1"
run_id="$2"
n_action_steps="${3:-20}"

case "$variant" in
  wristfront)
    train_run="act_stack_two_cubes_vertical_20ep_b8_30k_seed1000"
    cameras='{"front":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'
    required_video=(/dev/video0 /dev/video2)
    ;;
  wristonly)
    train_run="act_stack_two_cubes_vertical_20ep_wristonly_b8_30k_seed1000"
    cameras='{"wrist":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'
    required_video=(/dev/video0)
    ;;
  redleft)
    train_run="act_stack_two_cubes_redleft_20ep_b8_30k_seed1000"
    cameras='{"front":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'
    required_video=(/dev/video0 /dev/video2)
    ;;
  combined)
    train_run="act_stack_two_cubes_combined_40ep_b8_30k_seed1000"
    cameras='{"front":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'
    required_video=(/dev/video0 /dev/video2)
    ;;
  *)
    echo "VARIANT must be one of: wristfront, wristonly, redleft, combined." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! "$run_id" =~ ^eval_[a-zA-Z0-9._-]+$ ]]; then
  echo "RUN_ID must begin with eval_ and contain only letters, numbers, dot, underscore, or hyphen." >&2
  exit 2
fi
if [[ ! "$n_action_steps" =~ ^[0-9]+$ ]] || (( n_action_steps < 1 || n_action_steps > 100 )); then
  echo "N_ACTION_STEPS must be an integer from 1 to 100." >&2
  exit 2
fi

# Optional inference-mode overrides (do not change the trained weights):
#   EVAL_TEMPORAL_ENSEMBLE=<coeff>  smooth + reactive control (blends overlapping
#       action chunks every step); forces n_action_steps=1 as ACT requires. Try 0.01.
#   EVAL_EPISODE_TIME_S=<seconds>   longer horizon so multi-attempt recoveries can finish.
episode_time_s="${EVAL_EPISODE_TIME_S:-20}"
policy_ensemble_args=()
if [[ -n "${EVAL_TEMPORAL_ENSEMBLE:-}" ]]; then
  n_action_steps=1
  policy_ensemble_args+=("--policy.temporal_ensemble_coeff=$EVAL_TEMPORAL_ENSEMBLE")
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
record_bin="$(dirname -- "$python_bin")/lerobot-record"
model="$repo_root/outputs/train/$train_run/checkpoints/030000/pretrained_model"
dataset_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
dataset_root="$dataset_base/GY-William/$run_id"
latency_log="$repo_root/outputs/eval_latency/$run_id.csv"

export PYTHONNOUSERSITE=1

pose_cmd=(
  "$python_bin"
  "$script_dir/../common/check_start_pose.py"
  --profile vertical
)

# Scene gate: the wrist camera (video0) must see BOTH cubes fully in frame before
# the trial runs, so a clipped/missing cube can't silently confound the result.
# Set EVAL_SKIP_WRIST_CHECK=1 to bypass.
wristview_cmd=(
  "$python_bin"
  "$script_dir/../common/check_wrist_cube_view.py"
  --camera-index 0
)

record_cmd=(
  "$record_bin"
  --robot.type=so101_follower
  --robot.port=/dev/ttyACM0
  --robot.id=lerobot_follower_arm
  "--robot.cameras=$cameras"
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
  "--dataset.episode_time_s=$episode_time_s"
  --dataset.reset_time_s=0
  --dataset.fps=30
  --dataset.push_to_hub=false
  --display_data=false
  --play_sounds=false
  "--latency_log_path=$latency_log"
  --latency_warmup_frames=30
)
if (( ${#policy_ensemble_args[@]} > 0 )); then
  record_cmd+=("${policy_ensemble_args[@]}")
fi

# An interrupted run (Ctrl-C before the first episode is written) leaves a
# dataset skeleton with meta/info.json but no tasks.parquet. That stub holds no
# recorded data and is safe to clear, so aborting a trial does not burn the
# RUN_ID. Anything that already contains a parquet or mp4 may hold real episodes
# and is never touched.
eval_dataset_is_empty_stub() {
  local root="$1"
  if [[ -d "$root/data" || -d "$root/videos" ]]; then
    return 1
  fi
  if [[ -n "$(find "$root" -type f \( -name '*.parquet' -o -name '*.mp4' \) -print -quit)" ]]; then
    return 1
  fi
  return 0
}

if [[ -f "$dataset_root/meta/info.json" ]]; then
  if [[ ! -f "$dataset_root/meta/tasks.parquet" ]]; then
    if eval_dataset_is_empty_stub "$dataset_root"; then
      if "$dry_run"; then
        echo "would clear empty dataset stub left by an interrupted run: $dataset_root"
      else
        echo "Clearing empty dataset stub left by an interrupted run: $dataset_root"
        rm -rf -- "$dataset_root"
      fi
    else
      echo "Existing evaluation dataset is incomplete but not empty: $dataset_root" >&2
      echo "Use a new RUN_ID; this script will not delete data it did not verify as empty." >&2
      exit 1
    fi
  else
    record_cmd+=(--resume=true)
  fi
elif [[ -e "$dataset_root" ]]; then
  echo "Dataset path exists but has no complete metadata: $dataset_root" >&2
  echo "Use a new RUN_ID; this script will not delete the existing path." >&2
  exit 1
fi

if "$dry_run"; then
  printf 'variant: %s\nmodel: %s\n\npose check:\n  ' "$variant" "$train_run/030000"
  printf '%q ' "${pose_cmd[@]}"
  printf '\n\nwrist-view scene check:\n  '
  printf '%q ' "${wristview_cmd[@]}"
  printf '\n\nrecord command:\n  '
  printf '%q ' "${record_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$python_bin" "$record_bin" "$model/model.safetensors" /dev/ttyACM0 "${required_video[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

echo "Checking the vertical-gripper follower start pose..."
"${pose_cmd[@]}"

if [[ -z "${EVAL_SKIP_WRIST_CHECK:-}" ]]; then
  echo "Checking the wrist camera sees both cubes fully in frame..."
  "${wristview_cmd[@]}"
fi

echo
echo "Starting one vertical ACT trial ($variant)"
echo "  model: $train_run/030000"
case "$variant" in
  wristfront) echo "  cameras: front=/dev/video2, wrist=/dev/video0" ;;
  wristonly)  echo "  cameras: wrist=/dev/video0 (front camera not used)" ;;
esac
echo "  n_action_steps: $n_action_steps"
echo "  dataset: GY-William/$run_id"
echo "Keep hands clear and be ready to stop the robot."
"${record_cmd[@]}"
