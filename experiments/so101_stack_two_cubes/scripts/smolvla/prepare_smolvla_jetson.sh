#!/usr/bin/env bash
# Make a SmolVLA checkpoint trained on the A4500 loadable on the Jetson, then
# prove it loads and measure what it costs to run.
#
# Two things break a straight copy, and neither reports itself as a missing file:
#
#   1. The checkpoint's policy_preprocessor.json hardcodes tokenizer_name as the
#      Hub id HuggingFaceTB/SmolVLM2-500M-Video-Instruct. --policy.vlm_model_name
#      does not override it, this host has no route to huggingface.co, and the
#      processor files were fetched with local_dir=, which does not populate the
#      cache a repo-id lookup consults. This is what killed the first scheduled
#      training run on the other host. Fixed here without downloading anything:
#      the files already on this machine are copied into the hub cache layout and
#      refs/main is pinned to the same verified revision.
#
#   2. config.json records vlm_model_name as the absolute path it was trained
#      with, which exists only on the A4500. Rewritten to the local one.
#
# Then it loads the policy and, unless --no-bench, times forward passes. Latency
# has to be known before physical trials, not after: the Diffusion Policy
# comparison in this project was confounded exactly that way.
#
# Usage:  prepare_smolvla_jetson.sh CHECKPOINT_DIR [--no-bench] [--iters N]
#   CHECKPOINT_DIR  a .../checkpoints/030000/pretrained_model directory

set -uo pipefail

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }

bench=true
iters=50
ckpt=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-bench) bench=false; shift ;;
    --iters) iters="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) ckpt="$1"; shift ;;
  esac
done
[[ -n "$ckpt" ]] || { usage >&2; exit 2; }

python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
processor_dir="${SMOLVLA_BACKBONE_MODEL:-/home/hai/models/smolvlm2_500m_processor_7b375e1}"
backbone_repo="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
backbone_rev="${SMOLVLA_BACKBONE_REVISION:-7b375e1b73b11138ff12fe22c8f2822d8fe03467}"
hub_root="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"

say() { printf '[prepare] %s\n' "$*"; }
die() { printf '[prepare] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$ckpt" ]] || die "checkpoint directory not found: $ckpt"
for f in config.json model.safetensors policy_preprocessor.json; do
  [[ -f "$ckpt/$f" ]] || die "checkpoint is missing $f"
done
[[ -d "$processor_dir" ]] || die "processor directory not found: $processor_dir"
[[ -x "$python_bin" ]] || die "python not found: $python_bin"
say "checkpoint : $ckpt"
say "processor  : $processor_dir"

# --- 1. hub cache, built from files that are already here --------------------
snap="$hub_root/snapshots/$backbone_rev"
if [[ -f "$snap/config.json" && -f "$snap/tokenizer.json" ]]; then
  say "hub cache already present"
else
  say "building hub cache for $backbone_repo@${backbone_rev:0:7}"
  mkdir -p "$snap" "$hub_root/refs" || die "cannot write $hub_root"
  copied=0
  for f in added_tokens.json chat_template.json config.json generation_config.json \
           merges.txt preprocessor_config.json processor_config.json \
           special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
    if [[ -f "$processor_dir/$f" ]]; then
      cp -f "$processor_dir/$f" "$snap/$f" && copied=$((copied + 1))
    else
      say "  warning: $f absent from the processor directory"
    fi
  done
  (( copied > 0 )) || die "copied no processor files; is $processor_dir populated?"
  say "  copied $copied files"
fi
# A bare from_pretrained(repo_id) asks for "main"; pin it at the verified commit.
printf '%s' "$backbone_rev" > "$hub_root/refs/main"
say "refs/main -> ${backbone_rev:0:7}"

# --- 2. repoint vlm_model_name at this machine -------------------------------
PYTHONNOUSERSITE=1 "$python_bin" - "$ckpt" "$processor_dir" <<'PY' || die "could not update config.json"
import json, sys
from pathlib import Path

ckpt, processor = Path(sys.argv[1]), sys.argv[2]
cfg_path = ckpt / "config.json"
cfg = json.loads(cfg_path.read_text())
old = cfg.get("vlm_model_name")
if old == processor:
    print(f"[prepare] vlm_model_name already local: {processor}")
else:
    cfg["vlm_model_name"] = processor
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[prepare] vlm_model_name {old} -> {processor}")
PY

# --- 3. prove it loads, offline ----------------------------------------------
say "loading policy offline"
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 "$python_bin" - "$ckpt" "$bench" "$iters" <<'PY'
import sys, time
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

ckpt, bench, iters = sys.argv[1], sys.argv[2] == "true", int(sys.argv[3])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = SmolVLAPolicy.from_pretrained(ckpt)
policy.to(device).eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
cams = list(policy.config.image_features)
print(f"[prepare] loaded on {device}; cameras={cams}; "
      f"chunk={policy.config.chunk_size}; n_action_steps={policy.config.n_action_steps}")

if not bench:
    raise SystemExit(0)

# Synthetic observation with the shapes the policy declares. This measures the
# model, not the cameras; a real trial adds capture and USB latency on top.
state_dim = policy.config.robot_state_feature.shape[0]
obs = {k: torch.rand(3, 480, 640) for k in cams}
obs["observation.state"] = torch.rand(state_dim)
obs["task"] = "Stack the yellow cube on top of the red cube"

# The number that matters is the chunk refresh, not the cached replay, so measure
# them separately and take many samples of the expensive one. Resetting before
# every timed call forces a refresh each time.
def timed(call):
    t0 = time.perf_counter()
    with torch.inference_mode():
        call()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0

for _ in range(3):                       # absorb lazy init before anything is recorded
    policy.reset()
    timed(lambda: policy.select_action(pre(obs)))

refresh, cached = [], []
for _ in range(iters):
    policy.reset()
    refresh.append(timed(lambda: policy.select_action(pre(obs))))
    cached.append(timed(lambda: policy.select_action(pre(obs))))

def pct(xs, q):
    s = sorted(xs)
    return s[min(int(len(s) * q), len(s) - 1)]

n_act = policy.config.n_action_steps
budget = 1000.0 / 30.0
r50, r95 = pct(refresh, 0.5), pct(refresh, 0.95)
c50 = pct(cached, 0.5)
print(f"[prepare] chunk refresh over {len(refresh)} samples (ms): "
      f"p50={r50:.1f} p95={r95:.1f} min={min(refresh):.1f} max={max(refresh):.1f}")
print(f"[prepare] cached replay  over {len(cached)} samples (ms): p50={c50:.1f}")

# What a synchronous 30 FPS loop actually gets: one refresh, then n_act-1 replays.
cycle = r50 + (n_act - 1) * c50
motion = n_act * budget
print(f"[prepare] synchronous cycle: {n_act} actions cost {cycle:.0f} ms of wall time, "
      f"of which {r50:.0f} ms ({100.0 * r50 / cycle:.0f}%) is the robot standing still")
print(f"[prepare] effective rate {1000.0 * n_act / cycle:.1f} FPS against a 30 FPS target; "
      f"the arm moves for {motion:.0f} ms then stalls for {r50:.0f} ms")
print("[prepare] NOTE: this measures LeRobot's SYNCHRONOUS loop. lerobot.async_inference "
      "generates the next chunk while the current one executes and removes the stall "
      "without making the model faster.")
PY
status=$?
(( status == 0 )) || die "the policy did not load or benchmark cleanly (status $status)"
say "OK — checkpoint is ready for evaluation on this machine"
