#!/usr/bin/env bash

set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: prepare_act_v2_subsets.sh [--dry-run]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
dataset_repo="GY-William/lerobot_stack_two_cubes_v2"
dataset_root="${ACT_V2_DATASET_ROOT:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$dataset_repo}"
positions_csv="${ACT_V2_POSITIONS_CSV:-$experiment_dir/manifests/act_v2_start_positions.csv}"
manifest="${ACT_V2_MANIFEST:-$experiment_dir/manifests/act_v2_subsets.json}"

analyze_cmd=(
  "$python_bin" "$script_dir/../common/analyze_cube_placements.py"
  "--dataset-root=$dataset_root"
  "--repo-id=$dataset_repo"
  "--output=$positions_csv"
)
validate_cmd=(
  "$python_bin" "$script_dir/../common/validate_act_v2_positions.py"
  "$positions_csv" --expected-episodes=50
)
manifest_cmd=(
  "$python_bin" "$script_dir/generate_act_subsets.py"
  "$positions_csv"
  --sizes 30 50
  "--dataset-repo-id=$dataset_repo"
  "--output=$manifest"
)

if "$dry_run"; then
  printf 'start-frame analysis:\n  '
  printf '%q ' "${analyze_cmd[@]}"
  printf '\n\nquality gate:\n  '
  printf '%q ' "${validate_cmd[@]}"
  printf '\n\nsubset manifest:\n  '
  printf '%q ' "${manifest_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$python_bin" "$dataset_root/meta/info.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

episode_count="$($python_bin - "$dataset_root/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path

print(int(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["total_episodes"]))
PY
)"
if [[ "$episode_count" != 50 ]]; then
  echo "ACT v2 must contain exactly 50 episodes before subset preparation; found $episode_count." >&2
  exit 1
fi
if [[ -e "$positions_csv" || -e "$manifest" ]]; then
  echo "Refusing to overwrite existing v2 audit artifacts. Review or remove them explicitly first." >&2
  exit 1
fi

"${analyze_cmd[@]}"
"${validate_cmd[@]}"
"${manifest_cmd[@]}"
echo "ACT v2 nested 30/50-episode subsets are ready."
