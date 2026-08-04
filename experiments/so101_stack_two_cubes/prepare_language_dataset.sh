#!/usr/bin/env bash

set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: prepare_language_dataset.sh [--dry-run]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
edit_bin="$(dirname -- "$python_bin")/lerobot-edit-dataset"
dataset_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
first_repo="GY-William/lerobot_stack_two_cubes"
second_repo="GY-William/lerobot_stack_red_on_yellow"
merged_repo="GY-William/lerobot_stack_two_orders_language"
first_root="$dataset_base/$first_repo"
second_root="$dataset_base/$second_repo"
merged_root="$dataset_base/$merged_repo"

merge_cmd=(
  "$edit_bin"
  "--new_repo_id=$merged_repo"
  "--new_root=$merged_root"
  --operation.type=merge
  "--operation.repo_ids=['$first_repo','$second_repo']"
  "--operation.roots=['$first_root','$second_root']"
  --push_to_hub=false
)

validate_cmd=(
  "$python_bin" "$script_dir/validate_language_dataset.py" "$merged_root"
  --min-tasks=2
  --min-episodes-per-task=30
  '--expected-task=Stack the yellow cube on top of the red cube'
  '--expected-task=Stack the red cube on top of the yellow cube'
)

if "$dry_run"; then
  printf 'merge command:\n  '
  printf '%q ' "${merge_cmd[@]}"
  printf '\n\nvalidation command:\n  '
  printf '%q ' "${validate_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$edit_bin" "$first_root/meta/info.json" "$second_root/meta/info.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done
if [[ -e "$merged_root" ]]; then
  echo "Refusing to overwrite existing merged dataset: $merged_root" >&2
  exit 1
fi

"${merge_cmd[@]}"
"${validate_cmd[@]}"
