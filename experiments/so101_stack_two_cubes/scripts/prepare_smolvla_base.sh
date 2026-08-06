#!/usr/bin/env bash

set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: prepare_smolvla_base.sh [--dry-run]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
hf_bin="${HF_CLI:-/home/hai/.local/bin/hf}"
manifest="${SMOLVLA_BASE_MANIFEST:-$experiment_dir/manifests/smolvla_base.json}"
local_dir="${SMOLVLA_BASE_MODEL:-/home/hai/models/lerobot_smolvla_base_c83c316}"
backbone_dir="${SMOLVLA_BACKBONE_MODEL:-/home/hai/models/smolvlm2_500m_processor_7b375e1}"
endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
download_attempts="${SMOLVLA_DOWNLOAD_ATTEMPTS:-10}"
if [[ ! "$download_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMOLVLA_DOWNLOAD_ATTEMPTS must be a positive integer." >&2
  exit 2
fi

if [[ ! -f "$manifest" ]]; then
  echo "SmolVLA base manifest does not exist: $manifest" >&2
  exit 1
fi

mapfile -t manifest_values < <("$python_bin" -c '
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["repo_id"])
print(manifest["revision"])
for filename in manifest["required_files"]:
    print(filename)
' "$manifest")
repo_id="${manifest_values[0]}"
revision="${manifest_values[1]}"
files=("${manifest_values[@]:2}")
mapfile -t backbone_values < <("$python_bin" -c '
import json
import sys
backbone = json.load(open(sys.argv[1], encoding="utf-8"))["backbone"]
print(backbone["repo_id"])
print(backbone["revision"])
for filename in backbone["required_files"]:
    print(filename)
' "$manifest")
backbone_repo_id="${backbone_values[0]}"
backbone_revision="${backbone_values[1]}"
backbone_files=("${backbone_values[@]:2}")

download_cmd=(
  /usr/bin/env "HF_ENDPOINT=$endpoint"
  "$hf_bin" download "$repo_id"
  "${files[@]}"
  "--revision=$revision"
  "--local-dir=$local_dir"
  --max-workers=2
)
backbone_download_cmd=(
  /usr/bin/env "HF_ENDPOINT=$endpoint"
  "$hf_bin" download "$backbone_repo_id"
  "${backbone_files[@]}"
  "--revision=$backbone_revision"
  "--local-dir=$backbone_dir"
  --max-workers=2
)
verify_cmd=(
  "$python_bin" "$script_dir/verify_smolvla_base.py"
  "--root=$local_dir"
  "--backbone-root=$backbone_dir"
  "--manifest=$manifest"
)

if "$dry_run"; then
  printf 'download command:\n  '
  printf '%q ' "${download_cmd[@]}"
  printf '\n\nprocessor/config download command:\n  '
  printf '%q ' "${backbone_download_cmd[@]}"
  printf '\n\nverification command:\n  '
  printf '%q ' "${verify_cmd[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$python_bin" "$hf_bin"; do
  if [[ ! -x "$required_path" ]]; then
    echo "Required executable does not exist: $required_path" >&2
    exit 1
  fi
done

run_with_retries() {
  local label="$1"
  shift
  local attempt
  for ((attempt = 1; attempt <= download_attempts; attempt++)); do
    echo "$label download attempt $attempt/$download_attempts"
    if "$@"; then
      return 0
    fi
    if (( attempt < download_attempts )); then
      echo "$label download interrupted; retrying the retained partial download in 10 seconds." >&2
      sleep 10
    fi
  done
  echo "$label download failed after $download_attempts attempts." >&2
  return 1
}

mkdir -p "$local_dir" "$backbone_dir"
run_with_retries "SmolVLA base" "${download_cmd[@]}"
run_with_retries "SmolVLM2 processor/config" "${backbone_download_cmd[@]}"
"${verify_cmd[@]}"
