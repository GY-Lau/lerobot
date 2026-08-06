#!/usr/bin/env bash

set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -ne 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: discard_last_dataset_episode.sh [--dry-run] REPO_ID EXPECTED_EPISODE_COUNT" >&2
  exit 2
fi
repo_id="$1"
expected_count="$2"
if [[ ! "$repo_id" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "REPO_ID must have the form owner/name." >&2
  exit 2
fi

python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
edit_bin="$(dirname -- "$python_bin")/lerobot-edit-dataset"
dataset_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
dataset_root="$dataset_base/$repo_id"
info_path="$dataset_root/meta/info.json"

if [[ ! -f "$info_path" ]]; then
  echo "Dataset does not exist: $dataset_root" >&2
  exit 1
fi
episode_count="$($python_bin - "$info_path" <<'PY'
import json
import sys
from pathlib import Path

print(int(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["total_episodes"]))
PY
)"
if [[ "$episode_count" != "$expected_count" ]]; then
  echo "Refusing: expected $expected_count episodes, but dataset contains $episode_count." >&2
  exit 1
fi

last_index=$(( episode_count - 1 ))
timestamp="$(date +%Y%m%d_%H%M%S)"

if (( episode_count == 1 )); then
  backup_root="${dataset_root}_rejected_episode_${last_index}_${timestamp}"
  if "$dry_run"; then
    echo "would discard episode $last_index by moving the one-episode dataset to:"
    echo "  $backup_root"
    exit 0
  fi
  mv -- "$dataset_root" "$backup_root"
  echo "discarded episode $last_index; original data retained at: $backup_root"
  exit 0
fi

old_backup="${dataset_root}_old"
if [[ -e "$old_backup" ]]; then
  archived_backup="${dataset_root}_discard_backup_${timestamp}"
  if "$dry_run"; then
    echo "would archive existing edit backup: $old_backup -> $archived_backup"
  else
    mv -- "$old_backup" "$archived_backup"
  fi
fi

edit_cmd=(
  "$edit_bin"
  "--repo_id=$repo_id"
  "--root=$dataset_root"
  --operation.type=delete_episodes
  "--operation.episode_indices=[$last_index]"
  --push_to_hub=false
)

if "$dry_run"; then
  echo "would discard last episode index $last_index from $episode_count episodes:"
  printf '  %q ' "${edit_cmd[@]}"
  printf '\n'
  echo "the editor retains the original dataset at: $old_backup"
  exit 0
fi
if [[ ! -x "$edit_bin" ]]; then
  echo "Dataset editor does not exist or is not executable: $edit_bin" >&2
  exit 1
fi
"${edit_cmd[@]}"

new_count="$($python_bin - "$info_path" <<'PY'
import json
import sys
from pathlib import Path

print(int(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["total_episodes"]))
PY
)"
if (( new_count != episode_count - 1 )); then
  echo "Deletion verification failed: expected $(( episode_count - 1 )) episodes, found $new_count." >&2
  exit 1
fi
echo "discarded episode $last_index; dataset now contains $new_count episodes"
echo "pre-edit backup retained at: $old_backup"
