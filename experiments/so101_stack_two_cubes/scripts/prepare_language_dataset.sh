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
experiment_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
edit_bin="$(dirname -- "$python_bin")/lerobot-edit-dataset"
dataset_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
first_repo="${ACT_V2_DATASET_REPO:-GY-William/lerobot_stack_two_cubes_v2}"
second_repo="${INVERSE_DATASET_REPO:-GY-William/lerobot_stack_red_on_yellow}"
merged_repo="${LANGUAGE_DATASET_REPO:-GY-William/lerobot_stack_two_orders_language_v2}"
first_root="${ACT_V2_DATASET_ROOT:-$dataset_base/$first_repo}"
second_root="${INVERSE_DATASET_ROOT:-$dataset_base/$second_repo}"
merged_root="${LANGUAGE_DATASET_ROOT:-$dataset_base/$merged_repo}"
manifest="${ACT_V2_MANIFEST:-$experiment_dir/manifests/act_v2_subsets.json}"
subset_name="language30"
subset_repo="${first_repo}_${subset_name}"
subset_parent="${LANGUAGE_SUBSET_PARENT:-${first_root}_language_build}"
subset_root="$subset_parent/$subset_name"

if [[ ! -f "$manifest" ]]; then
  echo "ACT v2 subset manifest does not exist: $manifest" >&2
  exit 1
fi

episodes_json="$("$python_bin" -c '
import json
import sys

path, expected_repo = sys.argv[1:]
manifest = json.load(open(path, encoding="utf-8"))
if manifest.get("dataset_repo_id") != expected_repo:
    raise SystemExit(
        "Manifest repo_id {!r} does not match {!r}".format(
            manifest.get("dataset_repo_id"), expected_repo
        )
    )
episodes = manifest.get("subsets", {}).get("30", {}).get("episodes")
if not isinstance(episodes, list) or len(episodes) != 30:
    raise SystemExit("Manifest must contain exactly 30 episodes in subsets.30.episodes")
if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in episodes):
    raise SystemExit("Manifest episode indices must be non-negative integers")
if len(set(episodes)) != len(episodes):
    raise SystemExit("Manifest episode indices must be unique")
print(json.dumps(episodes, separators=(",", ":")))
' "$manifest" "$first_repo")"

split_cmd=(
  "$edit_bin"
  "--repo_id=$first_repo"
  "--root=$first_root"
  --operation.type=split
  "--operation.splits={\"$subset_name\":$episodes_json}"
  "--new_root=$subset_parent"
  --push_to_hub=false
)

merge_cmd=(
  "$edit_bin"
  "--new_repo_id=$merged_repo"
  "--new_root=$merged_root"
  --operation.type=merge
  "--operation.repo_ids=['$subset_repo','$second_repo']"
  "--operation.roots=['$subset_root','$second_root']"
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
  printf 'ACT v2 subset episodes (30): %s\n\n' "$episodes_json"
  printf 'split command:\n  '
  printf '%q ' "${split_cmd[@]}"
  printf '\n\nmerge command:\n  '
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
if [[ -e "$subset_root" ]]; then
  echo "Refusing to overwrite existing ACT v2 language subset: $subset_root" >&2
  exit 1
fi
if [[ -e "$merged_root" ]]; then
  echo "Refusing to overwrite existing merged dataset: $merged_root" >&2
  exit 1
fi

"${split_cmd[@]}"
"${merge_cmd[@]}"
"${validate_cmd[@]}"
