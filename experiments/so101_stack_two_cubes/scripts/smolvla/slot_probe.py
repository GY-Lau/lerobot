#!/usr/bin/env python3
"""Does SmolVLA's pretrained prior care which camera goes in which slot?

The released checkpoint declares observation.images.camera{1,2,3}. Per the
paper those slots were standardised during pretraining as top / wrist / side.
This project's fine-tune fed [wrist, front], so the wrist camera landed in the
slot pretraining used for a top view, and the front camera in the wrist slot.

If the backbone treats the slots interchangeably, permuting the assignment
should barely move the predicted action. If the slots carry position-dependent
meaning, permuting should move it about as much as showing a different scene
does -- which is the scale this compares against.

Flow matching is stochastic, so the noise is re-seeded identically before every
call; a same-vs-same control confirms that worked.
"""

import os
import sys

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

BASE = os.environ.get("SMOLVLA_BASE", "/home/ebots/guangyi_test_ws/models/lerobot_smolvla_base_c83c316")
ROOT = os.environ.get("SMOLVLA_PROBE_DATASET_ROOT",
                      "/home/ebots/guangyi_test_ws/datasets/GY-William/lerobot_stack_two_cubes_vertical_redleft_20ep")
REPO = "GY-William/lerobot_stack_two_cubes_vertical_redleft_20ep"
TASK = "Stack the yellow cube on top of the red cube"
SEED = 1000

device = torch.device("cuda")
# The released config sets load_vlm_weights=True, which makes transformers go to the
# Hub for backbone weights. Those weights are already inside smolvla_base's own
# model.safetensors, and this host has no route to huggingface.co, so point the
# backbone at the local processor directory and skip the separate fetch -- exactly
# what the training runs do.
from lerobot.configs.policies import PreTrainedConfig
cfg = PreTrainedConfig.from_pretrained(BASE)   # base class resolves the "type" discriminator
cfg.load_vlm_weights = False
cfg.vlm_model_name = "/home/ebots/guangyi_test_ws/models/smolvlm2_500m_processor_7b375e1"
cfg.device = "cuda"
policy = SmolVLAPolicy.from_pretrained(BASE, config=cfg)
policy.to(device).eval()
pre, _ = make_pre_post_processors(policy.config, pretrained_path=BASE)
slots = [k for k in policy.config.image_features]
print(f"base slots: {slots}")

ds = LeRobotDataset(REPO, root=ROOT)


def frame(i):
    it = ds[i]
    return (it["observation.images.wrist"], it["observation.images.front"], it["observation.state"])


def predict(wrist, front, state, mapping):
    """mapping: slot name -> image tensor (or None for an all-zero slot)."""
    obs = {"observation.state": state, "task": TASK}
    for s in slots:
        img = mapping.get(s)
        obs[s] = torch.zeros_like(wrist) if img is None else img
    batch = pre(obs)
    torch.manual_seed(SEED)          # identical flow-matching noise every call
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    return chunk[0].to("cpu", torch.float64).numpy()


def d(a, b):
    return float(np.abs(a - b).mean())


s1, s2, s3 = slots[0], slots[1], slots[2]
rows = []
for idx in (200, 2400, 5000, 7600, 10200):
    w, f, st = frame(idx)
    w, f, st = w.to(device), f.to(device), st.to(device)

    as_used = predict(w, f, st, {s1: w, s2: f})            # what training actually did
    control = predict(w, f, st, {s1: w, s2: f})            # same input, same seed
    swapped = predict(w, f, st, {s1: f, s2: w})            # cameras exchanged
    paper = predict(w, f, st, {s2: w, s3: f})              # wrist -> slot2, front -> slot3
    wrist_only = predict(w, f, st, {s1: w})                # front removed

    # scale reference: a genuinely different scene, same slot assignment
    w2, f2, st2 = frame((idx + 4000) % len(ds))
    other = predict(w2.to(device), f2.to(device), st2.to(device), {s1: w2.to(device), s2: f2.to(device)})

    rows.append((d(as_used, control), d(as_used, swapped), d(as_used, paper),
                 d(as_used, wrist_only), d(as_used, other)))
    print(f"  frame {idx:>6} done")

m = np.array(rows).mean(axis=0)
names = ["control (same input, same seed)", "cameras swapped", "paper mapping (wrist->slot2)",
         "front camera removed", "DIFFERENT SCENE (scale reference)"]
print("\n==== mean |Δaction| vs the assignment training used ====")
for n, v in zip(names, m):
    pct = 100.0 * v / m[4] if m[4] else float("nan")
    print(f"  {n:<34} {v:8.5f}   {pct:6.1f}% of a scene change")

print()
if m[0] > 1e-6:
    print("WARNING: control is non-zero, the noise was not held fixed; ratios are unreliable.")
elif m[1] < 0.05 * m[4]:
    print("VERDICT: slots look interchangeable to the backbone. The mismatch is not worth fixing.")
elif m[1] > 0.5 * m[4]:
    print("VERDICT: slot assignment matters about as much as the scene does. The fine-tune put")
    print("         both cameras in the wrong slots, and retraining with the paper mapping is a")
    print("         cheap thing to try.")
else:
    print("VERDICT: slots carry some meaning but less than the scene. Worth one retrain to check,")
    print("         not worth assuming it explains anything.")
