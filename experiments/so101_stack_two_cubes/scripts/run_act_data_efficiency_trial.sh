#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_act_data_efficiency_trial.sh [--dry-run] EPISODE_COUNT [USE_AMP]

EPISODE_COUNT must be 10, 20, or 30. Each invocation verifies the matched
30,000-step checkpoint and records exactly one 20-second physical trial.
Repeat the same command after resetting and logging each outcome.
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

episode_count="$1"
use_amp="${2:-false}"
case "$episode_count" in
  10|20)
    train_run="act_stack_two_cubes_${episode_count}ep_30k"
    ;;
  30)
    train_run="act_stack_two_cubes_30k"
    ;;
  *)
    echo "EPISODE_COUNT must be 10, 20, or 30." >&2
    exit 2
    ;;
esac

if [[ "$use_amp" != "true" && "$use_amp" != "false" ]]; then
  echo "USE_AMP must be true or false." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
checkpoint="$repo_root/outputs/train/$train_run/checkpoints/030000"
run_id="eval_act_${episode_count}ep_30k_protocol_v1"

verify_cmd=(
  "$python_bin"
  "$script_dir/verify_act_checkpoint.py"
  "$checkpoint"
  "--episode-count=$episode_count"
)
trial_cmd=(
  bash
  "$script_dir/run_act_trial.sh"
  "$run_id"
  030000
  "$use_amp"
  "$train_run"
)

if "$dry_run"; then
  printf 'checkpoint verification:\n  '
  printf '%q ' "${verify_cmd[@]}"
  printf '\n\n'
  "${trial_cmd[@]:0:2}" --dry-run "${trial_cmd[@]:2}"
  exit 0
fi

"${verify_cmd[@]}"
exec "${trial_cmd[@]}"
