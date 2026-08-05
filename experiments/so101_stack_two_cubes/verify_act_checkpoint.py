#!/usr/bin/env python3
"""Verify a completed ACT data-efficiency checkpoint and its training contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "train_config.json",
)


def verify_checkpoint(
    checkpoint_dir: Path,
    *,
    manifest_path: Path,
    episode_count: int,
    expected_steps: int = 30_000,
    expected_batch_size: int = 2,
    expected_seed: int = 1000,
) -> dict[str, object]:
    model_dir = checkpoint_dir / "pretrained_model"
    state_dir = checkpoint_dir / "training_state"
    for relative in REQUIRED_MODEL_FILES:
        path = model_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty checkpoint file: {path}")
    step_path = state_dir / "training_step.json"
    if not step_path.is_file():
        raise FileNotFoundError(f"Missing training state: {step_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        expected_episodes = manifest["subsets"][str(episode_count)]["episodes"]
    except KeyError as error:
        raise ValueError(f"Episode count {episode_count} is absent from {manifest_path}") from error

    train_config = json.loads((model_dir / "train_config.json").read_text(encoding="utf-8"))
    training_step = int(json.loads(step_path.read_text(encoding="utf-8"))["step"])
    configured_episodes = train_config["dataset"].get("episodes")
    if configured_episodes is None:
        # LeRobot serializes an omitted --dataset.episodes as null, meaning the
        # complete dataset rather than an empty subset. Resolve that shorthand
        # against the immutable subset manifest so the 30-episode baseline can
        # be checked by the same contract as the explicit 10/20 subsets.
        configured_episodes = list(range(int(manifest["total_episodes"])))
    else:
        configured_episodes = sorted(int(index) for index in configured_episodes)
    checks = {
        "training_step": (training_step, expected_steps),
        "configured_steps": (int(train_config["steps"]), expected_steps),
        "batch_size": (int(train_config["batch_size"]), expected_batch_size),
        "seed": (int(train_config["seed"]), expected_seed),
        "num_workers": (int(train_config["num_workers"]), 0),
        "dataset_repo_id": (
            train_config["dataset"]["repo_id"],
            manifest["dataset_repo_id"],
        ),
        "dataset_episodes": (
            configured_episodes,
            expected_episodes,
        ),
        "policy_type": (train_config["policy"]["type"], "act"),
        "use_amp": (bool(train_config["policy"]["use_amp"]), False),
        "optimizer_lr": (float(train_config["optimizer"]["lr"]), 1e-5),
        "scheduler": (train_config["scheduler"], None),
    }
    mismatches = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError("Checkpoint contract mismatch:\n" + json.dumps(mismatches, indent=2))

    return {
        "checkpoint": str(checkpoint_dir),
        "episode_count": episode_count,
        "steps": training_step,
        "model_size_bytes": (model_dir / "model.safetensors").stat().st_size,
        "episodes": expected_episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("act_data_subsets.json"))
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    result = verify_checkpoint(
        args.checkpoint_dir,
        manifest_path=args.manifest,
        episode_count=args.episode_count,
        expected_steps=args.steps,
        expected_batch_size=args.batch_size,
        expected_seed=args.seed,
    )
    print("ACT checkpoint: PASS")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
