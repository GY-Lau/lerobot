#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_act_v2_robust.sh [--dry-run] [RUN_NAME] [STEPS] [BATCH_SIZE] [SAVE_FREQ]

Train a fresh ACT v2 model on all 50 demonstrations with lightweight image
augmentation. Defaults: 60,000 updates, batch size 2, save every 10,000 steps.
The retained 30k baseline is never resumed or overwritten.
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

run_name="${1:-act_stack_two_cubes_v2_50ep_aug_60k}"
steps="${2:-60000}"
batch_size="${3:-2}"
save_freq="${4:-10000}"

# Preserve color identity: only mild illumination changes and a small affine
# perturbation are used. Affine covers camera/object translation, scale and a
# small rotation. At most two transforms are composed for any training image.
export ACT_V2_MAX_NUM_TRANSFORMS=2
export ACT_V2_IMAGE_TRANSFORMS_JSON='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"affine":{"weight":2.0,"type":"RandomAffine","kwargs":{"degrees":[-2.0,2.0],"translate":[0.04,0.04],"scale":[0.95,1.05]}}}'

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
