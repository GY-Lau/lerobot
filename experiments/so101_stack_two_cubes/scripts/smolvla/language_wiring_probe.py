#!/usr/bin/env python3
"""Does the instruction reach SmolVLA's action output at all?

The language experiment costs 30 new demonstrations and 40 physical trials
before it produces its first number. This asks the cheapest prerequisite
question first, on a checkpoint that already exists: when nothing changes but
the sentence, does the predicted action move?

Three quantities, in the same units, so the answer is a ratio rather than an
adjective:

  control       same observation, same sentence, same seed. Must be ~0, or the
                flow-matching noise was not held fixed and nothing below means
                anything.
  sentence      same observation, opposite colour order in the instruction.
  scene         different observation, same sentence. The scale reference --
                this is what "a real change" looks like for this policy.

Read it as sentence/scene:

  ~0            the instruction is not reaching the action head. A wiring bug,
                and worth finding before recording anything.
  small but >0  wired and being read, but this checkpoint saw one instruction
                in training, so it has no reason to act on it. Expected here,
                and the baseline the language-trained checkpoint must beat.
  comparable    the instruction moves the action as much as the scene does.

This does not measure whether the robot obeys -- that needs the physical
matrix. It measures whether obedience is even possible.

Usage:
  language_wiring_probe.py [CHECKPOINT_DIR] [DATASET_ROOT]
"""

import os
import sys

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

HOME = os.path.expanduser("~")
MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    f"{HOME}/lerobot/outputs/train/smolvla_expert_combined_40ep_b8_30k"
    "/checkpoints/030000/pretrained_model"
)
ROOT = sys.argv[2] if len(sys.argv) > 2 else (
    f"{HOME}/.cache/huggingface/lerobot/GY-William"
    "/lerobot_stack_two_cubes_vertical_redleft_20ep"
)
REPO = "GY-William/lerobot_stack_two_cubes_vertical_redleft_20ep"

TRAINED = "Stack the yellow cube on top of the red cube"
OPPOSITE = "Stack the red cube on top of the yellow cube"
PARAPHRASE = "Put the yellow block on the red block"
SEED = 1000
FRAMES = (200, 2400, 5000, 7600, 10200)

device = torch.device("cuda")
cfg = PreTrainedConfig.from_pretrained(MODEL)
cfg.device = "cuda"
policy = get_policy_class(cfg.type).from_pretrained(MODEL, config=cfg)
policy.to(device).eval()
pre, _ = make_pre_post_processors(policy.config, pretrained_path=MODEL)
slots = list(policy.config.image_features)
print(f"model : {MODEL}")
print(f"slots : {slots}\n")

ds = LeRobotDataset(REPO, root=ROOT)


def predict(item, task):
    obs = {"observation.state": item["observation.state"].to(device), "task": task}
    for slot in slots:
        obs[slot] = item[slot].to(device)
    batch = pre(obs)
    torch.manual_seed(SEED)  # identical flow-matching noise on every call
    with torch.inference_mode():
        return policy.predict_action_chunk(batch)[0].to("cpu", torch.float64).numpy()


def diff(a, b):
    return float(np.abs(a - b).mean())


rows = []
for idx in FRAMES:
    item = ds[idx]
    other = ds[(idx + 4000) % len(ds)]

    base = predict(item, TRAINED)
    rows.append((
        diff(base, predict(item, TRAINED)),     # control
        diff(base, predict(item, OPPOSITE)),    # opposite order
        diff(base, predict(item, PARAPHRASE)),  # same meaning, other words
        diff(base, predict(other, TRAINED)),    # different scene
    ))
    print(f"  frame {idx:>6} done")

names = ["control (same sentence)", "opposite colour order", "paraphrase, same meaning",
         "different scene (scale ref)"]
means = np.asarray(rows).mean(axis=0)
scene = means[3]

print("\n==== mean |delta action| vs the trained instruction ====")
for name, value in zip(names, means):
    share = 100.0 * value / scene if scene else float("nan")
    print(f"  {name:<30} {value:9.6f}   {share:6.1f}% of a scene change")

print()
if means[0] > 1e-6:
    print("WARNING: control is non-zero -- the noise was not held fixed; ratios are unreliable.")
elif means[1] < 1e-6:
    print("VERDICT: the instruction does not reach the action head at all. Fix this before")
    print("         recording the inverse task; no amount of data would make language work.")
elif means[1] < 0.05 * scene:
    print("VERDICT: wired and read, but barely acted on -- which is what a checkpoint trained")
    print("         on a single instruction should look like. This is the baseline; a")
    print("         two-order checkpoint has to move much further than this.")
else:
    print("VERDICT: the instruction already moves the action substantially. Worth re-checking")
    print("         that this checkpoint really only ever saw one instruction.")
