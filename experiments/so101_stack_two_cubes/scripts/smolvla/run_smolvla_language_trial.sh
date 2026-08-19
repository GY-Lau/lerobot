#!/usr/bin/env bash
# One SmolVLA language-control trial: same scene, one prompt, asynchronous.
#
# This is a thin wrapper around scripts/common/run_policy_trial.sh. Everything
# about driving the robot -- gates, cameras, controller, torque release -- lives
# there and is shared with every other physical trial in this project. What is
# specific to the language experiment, and therefore lives here, is the prompt
# for each condition and the 10-trial cap that keeps the 4x10 matrix balanced.
#
# Asynchronous, not synchronous. The earlier version of this script drove the
# robot through lerobot-record, the same synchronous loop under which this
# project measured SmolVLA at 0/5 -- the arm is frozen 43% of the time because
# a 1.255 s inference does not fit inside a 1.67 s chunk. Running the language
# matrix that way would have produced near-zero manipulation success in both
# color orders, and instruction-following cannot be scored at all when the robot
# never completes a stack either way. The 40 trials would have measured the
# controller.
#
# The consequence is that there is no recorded episode to review afterwards:
# the async client has no dataset code. Language trials are scored live, and
# log_smolvla_language_trial.py must be told so with --live-scored.
#
# Usage:
#   run_smolvla_language_trial.sh [--dry-run] CONDITION [TRAIN_RUN] [CHECKPOINT]
#
# CONDITION:
#   yellow_exact       Stack the yellow cube on top of the red cube
#   yellow_paraphrase  Put the yellow block on the red block
#   red_exact          Stack the red cube on top of the yellow cube
#   red_paraphrase     Put the red block on the yellow block
#
# The paraphrase conditions are what separate "learned the language" from
# "memorised one string", so they are not optional.
#
# Defaults:
#   TRAIN_RUN=smolvla_expert_two_orders_60ep_b8_30k   CHECKPOINT=030000
#
# That default is the released SmolVLA recipe at this project's training
# contract -- 30k steps, batch 8, seed 1000, expert_only -- so the language arm
# is comparable with smolvla_redleft and smolvla_combined. The older plan in
# docs/smolvla_peft.md named a rank-16 LoRA at 20k/batch 1; that would train
# fine but could not be set beside the other two arms.
#
# The horizon is pinned to 40 s, not the 20 s in docs/smolvla_peft.md. That 20 s
# predates the asynchronous controller and was chosen against ~20 s
# demonstrations; every SmolVLA arm actually evaluated in this project ran at
# 40 s. A horizon too short to finish a stack would depress manipulation success
# in both colour orders and confound the thing being measured, which is which
# order the robot chooses. Override with EVAL_EPISODE_TIME_S, but override it
# for every condition or the matrix is not balanced.
#
# Env: everything run_policy_trial.sh accepts.

set -uo pipefail

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

dry_run=false
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ $# -lt 1 || $# -gt 3 ]]; then usage >&2; exit 2; fi

condition="$1"
train_run="${2:-smolvla_expert_two_orders_60ep_b8_30k}"
checkpoint="${3:-030000}"

case "$condition" in
  yellow_exact)      prompt='Stack the yellow cube on top of the red cube' ;;
  yellow_paraphrase) prompt='Put the yellow block on the red block' ;;
  red_exact)         prompt='Stack the red cube on top of the yellow cube' ;;
  red_paraphrase)    prompt='Put the red block on the yellow block' ;;
  *)
    echo "Unknown CONDITION: $condition" >&2
    usage >&2
    exit 2 ;;
esac

if [[ ! "$train_run" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "TRAIN_RUN may contain only letters, numbers, dot, underscore and hyphen." >&2
  exit 2
fi
if [[ ! "$checkpoint" =~ ^[0-9]{6}$ ]]; then
  echo "CHECKPOINT must be six digits, for example 030000." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
trials_csv="$experiment_dir/results/smolvla_language_trials.csv"

# The cap used to count episodes in this condition's recorded dataset. Async
# records nothing, so it counts logged results instead -- which is the better
# source anyway: it caps what has actually been scored, not what was driven.
#
# Counted in awk, not python: the interpreter this project uses lives on the
# robot host, and when it is absent the count fails. A guard that fails open is
# worse than no guard -- it would have let an eleventh trial through in silence
# and unbalanced the matrix. awk is always there, and a failure to read a file
# that exists stops the trial below.
logged=0
if [[ -f "$trials_csv" ]]; then
  if ! logged="$(awk -F, -v want="$condition" '
      NR == 1 { for (i = 1; i <= NF; i++) if ($i == "condition") col = i; next }
      col && $col == want { n++ }
      END { print n + 0 }
    ' "$trials_csv")"; then
    echo "Could not read $trials_csv, so the 10-trial cap cannot be enforced." >&2
    exit 1
  fi
fi

if (( logged >= 10 )); then
  echo "Condition $condition already has $logged/10 logged trials; refusing trial 11." >&2
  echo "The 4x10 matrix is balanced by design -- an eleventh would unbalance it." >&2
  exit 1
fi

echo "language condition : $condition  (trial $((logged + 1)) of 10)"
echo "prompt             : $prompt"
echo

episode_time_s="${EVAL_EPISODE_TIME_S:-40}"
echo "horizon            : ${episode_time_s}s"
echo

cmd=(env "EVAL_TASK=$prompt" "EVAL_TRAIN_RUN=$train_run" "EVAL_CHECKPOINT=$checkpoint"
  "EVAL_EPISODE_TIME_S=$episode_time_s"
  bash "$script_dir/../common/run_policy_trial.sh")
"$dry_run" && cmd+=(--dry-run)
cmd+=(--async smolvla_language "eval_smolvla_$condition")

"${cmd[@]}"
status=$?

if (( status == 0 )) && ! "$dry_run"; then
  cat <<EOF

Score this trial by what you saw, then log it. There is no episode to re-watch.
Both questions are separate results:

  did it build a stable stack at all?      -> manipulation
  was it the colour order you asked for?   -> instruction following

A tidy stack in the wrong order is a language failure, not a success:

  python $script_dir/log_smolvla_language_trial.py --live-scored \\
    --condition $condition --observed-behavior <yellow_on_red|red_on_yellow|none> \\
    [--failure-label <label>] [--completion-time-s N]
EOF
fi
exit $status
