#!/usr/bin/env bash

# Run the teacher-forced action-error diagnostic on the vertical-gripper
# checkpoint, then print it side-by-side with the dual-camera baseline. This
# turns "did standing the gripper up restore grasp-height perception?" into
# numbers: overall / per-joint / phase-decile action error, in-distribution.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
python_bin="${LEROBOT_PYTHON:-/home/hai/miniconda3/envs/lerobot/bin/python}"
cache_base="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

vertical_run="${1:-act_stack_two_cubes_vertical_20ep_b8_30k_seed1000}"
stride="${DIAG_STRIDE:-2}"

vertical_model="$repo_root/outputs/train/$vertical_run/checkpoints/030000/pretrained_model"
vertical_repo="GY-William/lerobot_stack_two_cubes_vertical_20ep"
vertical_root="$cache_base/$vertical_repo"
vertical_out="$repo_root/outputs/diagnostics_vertical"
dualcam_summary="$repo_root/outputs/diagnostics_dualcam/teacher_forced_action_error_summary.json"

export PYTHONNOUSERSITE=1

if [[ ! -e "$vertical_model/model.safetensors" ]]; then
  echo "Missing vertical checkpoint: $vertical_model" >&2
  exit 1
fi

echo "Running teacher-forced diagnostic on the vertical checkpoint (stride $stride) ..."
"$python_bin" "$script_dir/diagnose_teacher_forced_action_error.py" \
  --model "$vertical_model" \
  --repo-id "$vertical_repo" \
  --root "$vertical_root" \
  --stride "$stride" \
  --out-dir "$vertical_out"

vertical_summary="$vertical_out/teacher_forced_action_error_summary.json"

echo
echo "==== vertical vs dual-camera (teacher-forced, in-distribution) ===="
"$python_bin" - "$vertical_summary" "$dualcam_summary" <<'PY'
import json, sys

v = json.load(open(sys.argv[1]))
d = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and __import__("os").path.exists(sys.argv[2]) else None

def line(label, vv, dd, fmt="{:>10}"):
    dcell = fmt.format(dd) if dd is not None else fmt.format("n/a")
    print(f"  {label:<26} {fmt.format(vv)}  {dcell}")

print(f"  {'metric':<26} {'vertical':>10} {'dualcam':>10}")
print("  " + "-" * 50)
line("overall 1-step MAE", v["overall_1step_mae_raw_units"],
     d and d["overall_1step_mae_raw_units"], "{:>10.3f}")
# Per-joint (union of keys, dualcam order).
joints = list(v["per_joint_1step_mae"].keys())
for j in joints:
    vv = v["per_joint_1step_mae"][j]
    dd = d["per_joint_1step_mae"].get(j) if d else None
    line(f"  {j}", vv, dd, "{:>10.3f}")
print("  phase-decile 1-step MAE (0=start .. 9=end):")
vph = v["phase_decile_mae"]
dph = d["phase_decile_mae"] if d else [None] * len(vph)
for i, (a, b) in enumerate(zip(vph, dph)):
    line(f"  d{i}", a, b, "{:>10.3f}")

if d is None:
    print("\n  (dual-camera baseline summary not found; showing vertical only)")
else:
    print("\n  Lower = the policy commands the expert action more accurately given the true state.")
    print("  Expect the biggest vertical improvement on wrist_flex and the grasp/place deciles")
    print("  IF the restored side-view actually gave back the grasp-height cue.")
PY
