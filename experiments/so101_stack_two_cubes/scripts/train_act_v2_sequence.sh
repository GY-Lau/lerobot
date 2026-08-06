#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_act_v2_sequence.sh [--dry-run] [RUN_30] [RUN_50]

Train and verify the ACT v2 30-episode checkpoint before starting the matched
50-episode checkpoint. A complete existing checkpoint is verified and reused;
an incomplete existing output causes the sequence to stop.
EOF
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

run_30="${1:-act_stack_two_cubes_v2_30ep_30k}"
run_50="${2:-act_stack_two_cubes_v2_50ep_30k}"
for run_name in "$run_30" "$run_50"; do
  if [[ ! "$run_name" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "Run names may contain only letters, numbers, dot, underscore, and hyphen." >&2
    exit 2
  fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
manifest="${ACT_V2_MANIFEST:-$script_dir/../manifests/act_v2_subsets.json}"

verify() {
  local episode_count="$1"
  local run_name="$2"
  "$python_bin" "$script_dir/verify_act_checkpoint.py" \
    "$repo_root/outputs/train/$run_name/checkpoints/030000" \
    --manifest "$manifest" \
    --episode-count "$episode_count"
}

run_or_verify() {
  local episode_count="$1"
  local run_name="$2"
  local output_dir="$repo_root/outputs/train/$run_name"

  if [[ -e "$output_dir" ]]; then
    echo "Existing output found; requiring a complete matched checkpoint: $output_dir"
    verify "$episode_count" "$run_name"
    return
  fi

  /bin/bash "$script_dir/train_act_v2.sh" "$episode_count" "$run_name"
  verify "$episode_count" "$run_name"
}

if "$dry_run"; then
  for specification in "30:$run_30" "50:$run_50"; do
    episode_count="${specification%%:*}"
    run_name="${specification#*:}"
    echo "=== ACT v2 $episode_count episodes ==="
    /bin/bash "$script_dir/train_act_v2.sh" --dry-run "$episode_count" "$run_name"
    printf '\nverification command:\n  '
    printf '%q ' \
      "$python_bin" "$script_dir/verify_act_checkpoint.py" \
      "$repo_root/outputs/train/$run_name/checkpoints/030000" \
      --manifest "$manifest" --episode-count "$episode_count"
    printf '\n\n'
  done
  printf 'artifact inventory command:\n  '
  printf '%q ' "$python_bin" "$script_dir/inventory_model_artifacts.py"
  printf '\n'
  exit 0
fi

run_or_verify 30 "$run_30"
run_or_verify 50 "$run_50"
"$python_bin" "$script_dir/inventory_model_artifacts.py"
echo "ACT v2 sequential 30/50 training and verification: PASS"
