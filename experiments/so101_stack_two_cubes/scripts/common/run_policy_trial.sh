#!/usr/bin/env bash
# One physical trial, for any policy family, under either controller.
#
# Replaces the ACT-only runner so that ACT and SmolVLA are evaluated through the
# same gates, the same start pose, the same horizon and the same task string. A
# comparison is only about the policy if everything else is held identical, and
# keeping two runners in step by hand is how that stops being true.
#
# Two controllers:
#
#   sync (default)  lerobot-record drives the robot and waits for each action
#                   chunk. Records the episode, so it can be reviewed later.
#
#   --async         lerobot.async_inference: a policy server generates the next
#                   chunk while the client is still executing the current one.
#                   It does NOT record a dataset -- the client has no dataset
#                   code at all -- so an async trial is scored by watching the
#                   robot, with no episode to re-check afterwards.
#
# Whether async helps is arithmetic, and the script prints it: a chunk of N
# actions consumed at F fps lasts N/F seconds, and that is all the time the
# server has to produce the next one. On the Jetson SmolVLA takes about 1.85 s
# per refresh against 1.67 s of runway at 30 fps, so it cannot quite keep up;
# at 25 fps the runway is 2.0 s and it can. Set EVAL_FPS accordingly.
#
# Usage:
#   run_policy_trial.sh [--dry-run] [--async] VARIANT RUN_ID [N_ACTION_STEPS]
#
# Env:
#   EVAL_FPS               control rate           (default 30)
#   EVAL_EPISODE_TIME_S    trial length           (default 40)
#   EVAL_CHUNK_THRESHOLD   async: request the next chunk once the remaining
#                          fraction of the queue drops to this (default 1.0,
#                          i.e. immediately). Anything lower spends part of the
#                          runway before asking: at 0.9 the client waits for 5
#                          of 50 actions to be consumed, which at 25 fps leaves
#                          1.8 s to cover a 1.85 s refresh, and the arm stalls.
#                          LeRobot's own default is 0.5.
#   EVAL_SERVER_PORT       async policy server port (default 8080)
#   EVAL_SKIP_WRIST_CHECK  set to 1 to skip the start-scene gate
#   EVAL_TASK              instruction handed to the policy (default: stack
#                          yellow on red). Only a VLA reads it; ACT ignores it.
#                          The language experiment sets it per condition.
#   EVAL_TRAIN_RUN         override the variant's training run directory
#   EVAL_CHECKPOINT        checkpoint step, six digits (default 030000)

set -uo pipefail

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

dry_run=false
use_async=false
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --async)   use_async=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ $# -lt 2 || $# -gt 3 ]]; then usage >&2; exit 2; fi

variant="$1"
run_id="$2"
n_action_steps="${3:-}"

if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, numbers, dot, underscore and hyphen." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
record_bin="${LEROBOT_RECORD_BIN:-$(dirname -- "$python_bin")/lerobot-record}"
fps="${EVAL_FPS:-30}"
episode_time_s="${EVAL_EPISODE_TIME_S:-40}"
chunk_threshold="${EVAL_CHUNK_THRESHOLD:-1.0}"
server_port="${EVAL_SERVER_PORT:-8080}"

# The task string is what a VLA reads. It is fixed for every arm of the
# ACT-vs-SmolVLA comparison, so it is a constant here -- but the language
# experiment varies it deliberately, one prompt per condition, and that is the
# whole point of that experiment. EVAL_TASK is how it does so.
TASK="${EVAL_TASK:-Stack the yellow cube on top of the red cube}"
CAMS_BOTH='{"front":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'
CAMS_WRIST='{"wrist":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}'

# variant -> policy family, training run, cameras, devices it needs
case "$variant" in
  wristfront)       policy_type=act;     train_run=act_stack_two_cubes_vertical_20ep_b8_30k_seed1000;          cameras="$CAMS_BOTH";  required_video=(/dev/video0 /dev/video2) ;;
  wristonly)        policy_type=act;     train_run=act_stack_two_cubes_vertical_20ep_wristonly_b8_30k_seed1000; cameras="$CAMS_WRIST"; required_video=(/dev/video0) ;;
  redleft)          policy_type=act;     train_run=act_stack_two_cubes_redleft_20ep_b8_30k_seed1000;            cameras="$CAMS_BOTH";  required_video=(/dev/video0 /dev/video2) ;;
  combined)         policy_type=act;     train_run=act_stack_two_cubes_combined_40ep_b8_30k_seed1000;           cameras="$CAMS_BOTH";  required_video=(/dev/video0 /dev/video2) ;;
  smolvla_redleft)  policy_type=smolvla; train_run=smolvla_expert_redleft_20ep_b8_30k;                          cameras="$CAMS_BOTH";  required_video=(/dev/video0 /dev/video2) ;;
  smolvla_combined) policy_type=smolvla; train_run=smolvla_expert_combined_40ep_b8_30k;                         cameras="$CAMS_BOTH";  required_video=(/dev/video0 /dev/video2) ;;
  smolvla_language) policy_type=smolvla; train_run=smolvla_expert_two_orders_60ep_b8_30k;                        cameras="$CAMS_BOTH";  required_video=(/dev/video0 /dev/video2) ;;
  *)
    echo "VARIANT must be one of: wristfront, wristonly, redleft, combined," >&2
    echo "                        smolvla_redleft, smolvla_combined, smolvla_language." >&2
    exit 2 ;;
esac

# Chunk length is a property of the trained policy, not of this script.
if [[ "$policy_type" == act ]]; then
  chunk_actions=100
else
  chunk_actions=50
fi
: "${n_action_steps:=$chunk_actions}"

train_run="${EVAL_TRAIN_RUN:-$train_run}"
checkpoint="${EVAL_CHECKPOINT:-030000}"
if [[ ! "$checkpoint" =~ ^[0-9]{6}$ ]]; then
  echo "EVAL_CHECKPOINT must be six digits, for example 030000." >&2; exit 2
fi
model="$repo_root/outputs/train/$train_run/checkpoints/$checkpoint/pretrained_model"
dataset_root="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/GY-William/$run_id"
latency_log="$repo_root/outputs/eval_latency/${run_id}.csv"

runway=$(awk "BEGIN{printf \"%.3f\", $n_action_steps / $fps}")
echo "variant=$variant  policy=$policy_type  controller=$([[ $use_async == true ]] && echo async || echo sync)"
echo "  model    : $model"
echo "  fps=$fps  n_action_steps=$n_action_steps  horizon=${episode_time_s}s"
echo "  task     : $TASK"
echo "  a chunk of $n_action_steps actions at $fps fps lasts ${runway}s -- that is the budget a"
echo "  refresh has to fit inside for the arm not to stall."
if [[ "$use_async" == true ]]; then
  echo "  async: server on 127.0.0.1:$server_port, request next chunk at queue<=${chunk_threshold}"
  echo "  NOTE: the async client records nothing. Score this trial by watching the robot."
fi

# --- gates -------------------------------------------------------------------
gate_pose=(env PYTHONNOUSERSITE=1 "$python_bin" "$script_dir/check_start_pose.py" --profile vertical)
gate_scene=(env PYTHONNOUSERSITE=1 "$python_bin" "$script_dir/check_wrist_cube_view.py" --camera-index 0)

if "$dry_run"; then
  printf '\npose gate:\n  '; printf '%q ' "${gate_pose[@]}"
  printf '\nscene gate:\n  '; printf '%q ' "${gate_scene[@]}"
  printf '\n\ncontrol command:\n  '
fi

for dev in "${required_video[@]}"; do
  [[ -e "$dev" ]] || { echo "Camera device missing: $dev" >&2; "$dry_run" || exit 1; }
done
[[ -d "$model" ]] || { echo "Checkpoint not found: $model" >&2; "$dry_run" || exit 1; }

if ! "$dry_run"; then
  "${gate_pose[@]}" || { echo "Start pose gate failed; reposition the follower." >&2; exit 1; }
  if [[ "${EVAL_SKIP_WRIST_CHECK:-0}" != "1" ]]; then
    "${gate_scene[@]}" || { echo "Start scene gate failed; both cubes must be fully in the wrist view." >&2; exit 1; }
  fi
fi

# --- controller --------------------------------------------------------------
if [[ "$use_async" == true ]]; then
  # The async stack is an optional dependency group, so a machine that trains and
  # evaluates fine can still be missing it. Say so here rather than letting the
  # server exit on an import and look like a crash.
  # --dry-run only prints the commands, so it must not require the runtime the
  # commands would need. Checking here made the dry run unusable on any machine
  # that is not the robot host -- which is exactly where you preview a command.
  if ! "$dry_run" && ! PYTHONNOUSERSITE=1 "$python_bin" -c 'import grpc' >/dev/null 2>&1; then
    echo "The asynchronous controller needs grpcio, which is not installed." >&2
    echo "Install the wheel without disturbing anything else:" >&2
    echo "  $python_bin -m pip install --only-binary=:all: --no-deps grpcio==1.73.1" >&2
    echo "(lerobot also pins protobuf<6.32.0 in that group; the newer runtime this" >&2
    echo " machine already has works, and downgrading it would break wandb.)" >&2
    exit 1
  fi

  # PYTHONNOUSERSITE is not optional on the Jetson: ~/.local carries a newer
  # transformers than the conda environment, and under it lerobot's groot config
  # fails to even define itself, so importing the policy factory raises. Every
  # other entry point in this project sets it; these two were missed.
  # HF_HUB_OFFLINE matters as much as PYTHONNOUSERSITE here: the checkpoint's
  # preprocessor names the backbone by Hub id, and without it from_pretrained
  # goes to huggingface.co, which this machine cannot reach, and blocks there.
  server_cmd=(env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 "$python_bin" "$script_dir/async_policy_server.py"
    --host=127.0.0.1 "--port=$server_port" "--fps=$fps")
  # Not "-m lerobot.async_inference.robot_client": that module leaves the robot and
  # camera registries empty, so draccus rejects every --robot.type. The wrapper
  # imports them first and then calls the same entry point.
  client_cmd=(env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 "$python_bin" "$script_dir/async_robot_client.py"
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=lerobot_follower_arm
    "--robot.cameras=$cameras"
    "--task=$TASK"
    "--server_address=127.0.0.1:$server_port"
    "--policy_type=$policy_type"
    "--pretrained_name_or_path=$model"
    --policy_device=cuda --client_device=cpu
    "--actions_per_chunk=$n_action_steps"
    "--chunk_size_threshold=$chunk_threshold"
    "--fps=$fps")

  # An async trial writes no dataset, so these two logs are the only per-run
  # artifact it leaves. Name them before the dry run exits, so a preview says
  # where its output would land -- and so RUN_ID is visible at all, which under
  # the synchronous controller it was via --dataset.repo_id.
  server_log="$repo_root/outputs/eval_latency/${run_id}_server.log"
  client_log="$repo_root/outputs/eval_latency/${run_id}_client.log"

  if "$dry_run"; then
    printf '%q ' "${server_cmd[@]}"; printf '\n\nthen:\n  '
    printf '%q ' "${client_cmd[@]}"; printf '\n'
    printf '\nlogs:\n  %s\n  %s\n' "$server_log" "$client_log"
    exit 0
  fi

  mkdir -p "$repo_root/outputs/eval_latency"

  port_open() {
    env PYTHONNOUSERSITE=1 "$python_bin" -c "
import socket,sys
s=socket.socket(); s.settimeout(0.5)
sys.exit(0 if s.connect_ex(('127.0.0.1',$server_port))==0 else 1)"
  }

  # A server already listening is reused and left running. It keeps the
  # checkpoint in memory, so the next trial starts moving in seconds instead of
  # waiting out another load.
  server_pid=""
  if port_open; then
    echo "reusing the policy server already on 127.0.0.1:$server_port"
  else
    "${server_cmd[@]}" > "$server_log" 2>&1 &
    server_pid=$!
    trap 'kill "$server_pid" 2>/dev/null' EXIT INT TERM
    for _ in $(seq 1 30); do port_open && break; sleep 1; done
    kill -0 "$server_pid" 2>/dev/null || { echo "Policy server died; see $server_log" >&2; exit 1; }
    echo "policy server started (pid $server_pid, log $server_log)"
    echo "  it stops with this trial. To keep it warm across trials, start it yourself:"
    echo "    env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 $python_bin $script_dir/async_policy_server.py --host=127.0.0.1 --port=$server_port --fps=$fps &"
  fi
  # The server loads the checkpoint when the client first connects; SmolVLA took
  # 39 s of it on the Jetson. Timing the trial from process start charges that to
  # the robot and makes every trial a different length, which is not comparable
  # with the fixed-horizon ACT trials. Wait for the control loop to announce
  # itself, then start the clock.
  "${client_cmd[@]}" > "$client_log" 2>&1 &
  client_pid=$!
  trap 'kill "$client_pid" 2>/dev/null; [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null' EXIT INT TERM

  for _ in $(seq 1 300); do
    grep -q "Control loop thread starting" "$client_log" 2>/dev/null && break
    kill -0 "$client_pid" 2>/dev/null || break
    sleep 1
  done
  if ! grep -q "Control loop thread starting" "$client_log" 2>/dev/null; then
    echo "The control loop never started; see $client_log and $server_log" >&2
    exit 1
  fi
  echo "robot is moving -- ${episode_time_s}s trial starts now"
  sleep "$episode_time_s"
  kill "$client_pid" 2>/dev/null; sleep 2
  [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null

  # A killed client never reaches its disconnect, so disable_torque_on_disconnect
  # never fires and the arm is left rigid -- it cannot be repositioned by hand for
  # the next trial. Release it here, whatever the client did on the way out.
  echo "releasing motor torque"
  env PYTHONNOUSERSITE=1 "$python_bin" "$script_dir/release_torque.py" /dev/ttyACM0 \
    || echo "  WARNING: could not release torque; run $script_dir/release_torque.py by hand" >&2

  echo "Trial finished after ${episode_time_s}s of motion. Nothing was recorded;"
  echo "log the outcome with common/log_trial.py."
  exit 0
fi

# sync: lerobot-record owns the loop and writes the episode
if [[ -f "$dataset_root/meta/info.json" && ! -f "$dataset_root/meta/tasks.parquet" ]]; then
  if [[ -z "$(find "$dataset_root" -type f \( -name '*.parquet' -o -name '*.mp4' \) -print -quit)" \
        && ! -d "$dataset_root/data" && ! -d "$dataset_root/videos" ]]; then
    echo "Clearing empty dataset stub left by an interrupted run: $dataset_root"
    "$dry_run" || rm -rf -- "$dataset_root"
  else
    echo "Existing evaluation dataset is incomplete but not empty: $dataset_root" >&2
    echo "Use a new RUN_ID; this script will not delete data it did not verify as empty." >&2
    exit 1
  fi
fi

# HF_HUB_OFFLINE for the same reason as the async path: the checkpoint names
# its backbone by Hub id, so without it every trial makes a network round trip
# for a config that is already on disk.
record_cmd=(env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 "$record_bin"
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=lerobot_follower_arm
  "--robot.cameras=$cameras"
  "--policy.type=$policy_type"
  "--policy.pretrained_path=$model"
  --policy.device=cuda
  "--policy.n_action_steps=$n_action_steps"
  --policy.push_to_hub=false
  "--dataset.repo_id=GY-William/$run_id"
  "--dataset.root=$dataset_root"
  "--dataset.single_task=$TASK"
  --dataset.num_episodes=1
  "--dataset.episode_time_s=$episode_time_s"
  --dataset.reset_time_s=0
  "--dataset.fps=$fps"
  --dataset.push_to_hub=false
  --display_data=false --play_sounds=false
  "--latency_log_path=$latency_log")
[[ "$policy_type" == act ]] && record_cmd+=(--policy.use_amp=false)
[[ -f "$dataset_root/meta/tasks.parquet" ]] && record_cmd+=(--resume=true)

if "$dry_run"; then
  printf '%q ' "${record_cmd[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$repo_root/outputs/eval_latency"
"${record_cmd[@]}"
