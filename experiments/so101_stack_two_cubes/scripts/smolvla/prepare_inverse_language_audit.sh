#!/usr/bin/env bash

set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: prepare_inverse_language_audit.sh [--dry-run]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
dataset_repo="${INVERSE_DATASET_REPO:-GY-William/lerobot_stack_red_on_yellow}"
dataset_root="${INVERSE_DATASET_ROOT:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$dataset_repo}"
inverse_positions="${INVERSE_POSITIONS_CSV:-$experiment_dir/manifests/inverse_language_start_positions.csv}"
act_v2_positions="${ACT_V2_POSITIONS_CSV:-$experiment_dir/manifests/act_v2_start_positions.csv}"
act_v2_manifest="${ACT_V2_MANIFEST:-$experiment_dir/manifests/act_v2_subsets.json}"

analyze_cmd=(
  "$python_bin" "$script_dir/../common/analyze_cube_placements.py"
  "--dataset-root=$dataset_root"
  "--repo-id=$dataset_repo"
  "--output=$inverse_positions"
)
quality_cmd=(
  "$python_bin" "$script_dir/../common/validate_act_v2_positions.py"
  "$inverse_positions" --expected-episodes=30
)
balance_cmd=(
  "$python_bin" "$script_dir/validate_language_position_balance.py"
  "$act_v2_positions" "$act_v2_manifest" "$inverse_positions"
)

if "$dry_run"; then
  printf 'inverse start-frame analysis:\n  '
  printf '%q ' "${analyze_cmd[@]}"
  printf '\n\ninverse quality gate:\n  '
  printf '%q ' "${quality_cmd[@]}"
  printf '\n\ncross-task role-aligned balance gate:\n  '
  printf '%q ' "${balance_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in \
  "$python_bin" \
  "$dataset_root/meta/info.json" \
  "$act_v2_positions" \
  "$act_v2_manifest"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done
episode_count="$("$python_bin" -c '
import json
import sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["total_episodes"]))
' "$dataset_root/meta/info.json")"
if [[ "$episode_count" != 30 ]]; then
  echo "Inverse language dataset must contain exactly 30 episodes; found $episode_count." >&2
  exit 1
fi
if [[ -e "$inverse_positions" ]]; then
  echo "Refusing to overwrite existing inverse-task audit: $inverse_positions" >&2
  exit 1
fi

"${analyze_cmd[@]}"
"${quality_cmd[@]}"
"${balance_cmd[@]}"
echo "Inverse-task visual quality and cross-task position balance: PASS"
