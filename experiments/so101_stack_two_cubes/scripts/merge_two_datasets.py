#!/usr/bin/env python3
"""Merge two local LeRobot datasets into one (wrapper over lerobot merge_datasets).

Used to build the 40-episode combined set from the previous vertical 20 plus the
new red-left 20, so a single model can train on both. Episodes are concatenated
and re-indexed; both inputs must share the same features (cameras, state, action).
"""

from __future__ import annotations

import argparse

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-a", required=True)
    p.add_argument("--root-a", required=True)
    p.add_argument("--repo-b", required=True)
    p.add_argument("--root-b", required=True)
    p.add_argument("--out-repo", required=True)
    p.add_argument("--out-root", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ds_a = LeRobotDataset(args.repo_a, root=args.root_a)
    ds_b = LeRobotDataset(args.repo_b, root=args.root_b)
    print(f"A: {ds_a.meta.total_episodes} eps / {ds_a.meta.total_frames} frames  cams={ds_a.meta.video_keys}")
    print(f"B: {ds_b.meta.total_episodes} eps / {ds_b.meta.total_frames} frames  cams={ds_b.meta.video_keys}")
    if set(ds_a.meta.video_keys) != set(ds_b.meta.video_keys):
        raise SystemExit(f"Camera mismatch: {ds_a.meta.video_keys} vs {ds_b.meta.video_keys}")
    merged = merge_datasets([ds_a, ds_b], output_repo_id=args.out_repo, output_dir=args.out_root)
    print(f"MERGED: {merged.meta.total_episodes} eps / {merged.meta.total_frames} frames  -> {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
