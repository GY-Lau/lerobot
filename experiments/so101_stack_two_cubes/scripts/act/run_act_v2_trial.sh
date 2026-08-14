#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_act_v2_trial.sh [--dry-run] EPISODE_COUNT [USE_AMP]

EPISODE_COUNT must be 30 or 50. The matched final checkpoint and exact ACT v2
subset contract are verified before one 20-second physical trial is recorded.
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
  30|50) ;;
  *)
    echo "EPISODE_COUNT must be 30 or 50." >&2
    exit 2
    ;;
esac
if [[ "$use_amp" != "true" && "$use_amp" != "false" ]]; then
  echo "USE_AMP must be true or false." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
manifest="${ACT_V2_MANIFEST:-$experiment_dir/manifests/act_v2_subsets.json}"
train_run="act_stack_two_cubes_v2_${episode_count}ep_30k"
checkpoint="$repo_root/outputs/train/$train_run/checkpoints/030000"
run_id="eval_act_v2_${episode_count}ep_30k_protocol_v1"

verify_cmd=(
  "$python_bin"
  "$script_dir/verify_act_checkpoint.py"
  "$checkpoint"
  "--manifest=$manifest"
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
