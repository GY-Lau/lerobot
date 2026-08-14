#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_act_v2_photometric.sh [--dry-run] [RUN_NAME] [STEPS] [BATCH_SIZE] [SAVE_FREQ]

Train ACT v2 on all 50 demonstrations using brightness and contrast jitter only.
No transform changes image geometry. Defaults: 30,000 updates, batch size 2,
save every 5,000 steps. ACT_V2_SEED defaults to 1000.
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

run_name="${1:-act_stack_two_cubes_v2_50ep_photo_30k_seed1000}"
steps="${2:-30000}"
batch_size="${3:-2}"
save_freq="${4:-5000}"

# Photometric transforms preserve the pixel-space location of both cubes, so
# the observation remains geometrically consistent with the recorded action.
export ACT_V2_MAX_NUM_TRANSFORMS=2
export ACT_V2_IMAGE_TRANSFORMS_JSON='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}}}'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmd=(
  "$script_dir/train_act_v2.sh"
  50
  "$run_name"
  "$steps"
  "$batch_size"
  "$save_freq"
)
if "$dry_run"; then
  cmd=("$script_dir/train_act_v2.sh" --dry-run "${cmd[@]:1}")
fi

exec "${cmd[@]}"
