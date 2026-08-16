#!/usr/bin/env bash
# Unattended SmolVLA LoRA run for the shared A4500 host.
#
# The box is shared and its disk runs close to full, so this refuses to start
# rather than degrade someone else's job or fill the filesystem. It also proves
# the pipeline with a short smoke run before committing to the full one: an
# unattended job whose first execution is also its first test tends to waste the
# whole window it was scheduled for.
#
# Sequence: wait for a free GPU -> check disk -> verify the pinned base
# checkpoint -> smoke run -> full run. Any failed gate aborts with a logged
# reason and touches nothing.
#
# Usage:  scheduled_smolvla_run.sh [RUN_NAME]
# Env:    SMOLVLA_MIN_FREE_MIB   GPU free memory required        (default 14000)
#         SMOLVLA_MAX_GPU_UTIL   GPU utilisation allowed         (default 25)
#         SMOLVLA_MIN_DISK_GB    filesystem headroom required    (default 20)
#         SMOLVLA_WAIT_MINUTES   how long to keep retrying       (default 360)
#         SMOLVLA_STEPS          optimizer steps for the full run(default 20000)

set -uo pipefail

check_only=false
if [[ "${1:-}" == "--check-only" ]]; then
  check_only=true
  shift
fi

run_name="${1:-smolvla_lora_redleft_20ep_r16_20k}"
min_free_mib="${SMOLVLA_MIN_FREE_MIB:-14000}"
max_util="${SMOLVLA_MAX_GPU_UTIL:-25}"
min_disk_gb="${SMOLVLA_MIN_DISK_GB:-20}"
wait_minutes="${SMOLVLA_WAIT_MINUTES:-360}"
steps="${SMOLVLA_STEPS:-20000}"

ws=/home/ebots/guangyi_test_ws
repo="$ws/lerobot"
script_dir="$repo/experiments/so101_stack_two_cubes/scripts/smolvla"
log_dir="$ws/logs"
mkdir -p "$log_dir"
log="$log_dir/scheduled_smolvla_$(date +%Y%m%d_%H%M%S).log"

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$log"; }
abort() { say "ABORT: $*"; exit 1; }

say "run_name=$run_name steps=$steps host=$(hostname)"

# --- gate 1: a GPU nobody else is using -------------------------------------
gpu=""
deadline=$(( $(date +%s) + wait_minutes * 60 ))
while :; do
  while IFS=, read -r idx used total util; do
    idx="${idx// /}"; used="${used// /}"; total="${total// /}"; util="${util// /}"
    free=$(( total - used ))
    if (( free >= min_free_mib && util <= max_util )); then
      gpu="$idx"
      say "selected GPU $idx (${free} MiB free, ${util}% util)"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader,nounits)
  [[ -n "$gpu" ]] && break
  (( $(date +%s) >= deadline )) && abort "no GPU with ${min_free_mib} MiB free and <=${max_util}% util within ${wait_minutes} min"
  say "no free GPU; retrying in 20 min"
  sleep 1200
done

# --- gate 2: filesystem headroom --------------------------------------------
free_gb=$(df -BG --output=avail "$ws" | tail -1 | tr -dc '0-9')
(( free_gb >= min_disk_gb )) || abort "only ${free_gb} GB free on $ws, need ${min_disk_gb} GB"
say "disk OK: ${free_gb} GB free"

# --- shared training environment --------------------------------------------
export LEROBOT_PYTHON="$ws/venvs/lerobot-a4500-py310/bin/python"
export SMOLVLA_BASE_MODEL="$ws/models/lerobot_smolvla_base_c83c316"
export SMOLVLA_BACKBONE_MODEL="$ws/models/smolvlm2_500m_processor_7b375e1"
export SMOLVLA_BASE_MANIFEST="$repo/experiments/so101_stack_two_cubes/manifests/smolvla_base.json"
export LANGUAGE_DATASET_REPO=GY-William/lerobot_stack_two_cubes_vertical_redleft_20ep
export LANGUAGE_DATASET_ROOT="$ws/datasets/GY-William/lerobot_stack_two_cubes_vertical_redleft_20ep"
# Single-task dataset: this is the ACT-matched data comparison, not the
# two-order language experiment the gate defaults to.
export SMOLVLA_MIN_TASKS=1
export SMOLVLA_MIN_EPISODES_PER_TASK=20
export SMOLVLA_EXPECTED_TASKS="Stack the yellow cube on top of the red cube"
export LEROBOT_NUM_WORKERS="${LEROBOT_NUM_WORKERS:-8}"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONNOUSERSITE=1
# --policy.vlm_model_name redirects the backbone weights, but the base
# checkpoint's policy_preprocessor.json carries its own hardcoded
# "tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" -- a Hub id,
# not a path. This host cannot reach huggingface.co, so that lookup has to be
# served from the local hub cache. Downloading with local_dir= does NOT populate
# that cache, which is why the first scheduled attempt died here. Fail loudly
# rather than hang on a DNS timeout if the cache is ever missing.
export HF_HUB_OFFLINE=1

# --- gate 3: the pinned base checkpoint is intact ----------------------------
say "verifying pinned SmolVLA base"
"$LEROBOT_PYTHON" "$script_dir/verify_smolvla_base.py" \
  --root="$SMOLVLA_BASE_MODEL" --backbone-root="$SMOLVLA_BACKBONE_MODEL" \
  --manifest="$SMOLVLA_BASE_MANIFEST" \
  >>"$log" 2>&1 || abort "base checkpoint failed sha256 verification"

if "$check_only"; then
  say "check-only: resolved training command follows, nothing was started"
  bash "$script_dir/train_smolvla_peft.sh" --dry-run "$run_name" "$steps" 1 16 5000 bf16 2>&1 | tee -a "$log"
  exit 0
fi

# --- gate 4: smoke run -------------------------------------------------------
smoke="${run_name}_smoke"
rm -rf "$repo/outputs/train/$smoke"
say "smoke run: 20 steps"
if ! bash "$script_dir/train_smolvla_peft.sh" "$smoke" 20 1 16 20 bf16 >>"$log" 2>&1; then
  abort "smoke run failed; full run not started (see $log)"
fi
[[ -d "$repo/outputs/train/$smoke/checkpoints" ]] || abort "smoke run wrote no checkpoint"
say "smoke run OK"
rm -rf "$repo/outputs/train/$smoke"

# --- full run ----------------------------------------------------------------
say "starting full run: $steps steps on GPU $gpu"
bash "$script_dir/train_smolvla_peft.sh" "$run_name" "$steps" 1 16 5000 bf16 >>"$log" 2>&1
status=$?
if (( status == 0 )); then
  say "DONE: $repo/outputs/train/$run_name"
else
  say "FAILED with status $status"
fi
exit "$status"
