#!/usr/bin/env python3
"""Inventory trained policy artifacts and emit reproducible content hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    train_run: str
    unique_episodes: int


ARTIFACTS = (
    ArtifactSpec("act_10ep_30k", "act_stack_two_cubes_10ep_30k", 10),
    ArtifactSpec("act_20ep_30k", "act_stack_two_cubes_20ep_30k", 20),
    ArtifactSpec("act_30ep_30k", "act_stack_two_cubes_30k", 30),
    ArtifactSpec("diffusion_30ep_30k", "diffusion_stack_two_cubes_30k", 30),
    ArtifactSpec("act_v2_30ep_30k", "act_stack_two_cubes_v2_30ep_30k", 30),
    ArtifactSpec("act_v2_50ep_30k", "act_stack_two_cubes_v2_50ep_30k", 50),
)
FIELDNAMES = (
    "artifact_id",
    "policy_type",
    "unique_episodes",
    "optimizer_steps",
    "batch_size",
    "use_amp",
    "checkpoint_path",
    "model_size_bytes",
    "model_sha256",
    "train_config_sha256",
    "hub_repo_id",
    "publication_status",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifact(repo_root: Path, spec: ArtifactSpec) -> dict[str, object]:
    checkpoint = repo_root / "outputs" / "train" / spec.train_run / "checkpoints" / "030000"
    model_dir = checkpoint / "pretrained_model"
    state_dir = checkpoint / "training_state"
    required = {
        "model": model_dir / "model.safetensors",
        "model_config": model_dir / "config.json",
        "train_config": model_dir / "train_config.json",
        "training_step": state_dir / "training_step.json",
    }
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {label}: {path}")

    model_config = json.loads(required["model_config"].read_text(encoding="utf-8"))
    train_config = json.loads(required["train_config"].read_text(encoding="utf-8"))
    training_step = int(json.loads(required["training_step"].read_text(encoding="utf-8"))["step"])
    configured_steps = int(train_config["steps"])
    if training_step != configured_steps:
        raise ValueError(
            f"Incomplete checkpoint for {spec.artifact_id}: step {training_step}, configured {configured_steps}"
        )

    try:
        checkpoint_path = str(checkpoint.relative_to(repo_root))
    except ValueError:
        checkpoint_path = str(checkpoint)
    return {
        "artifact_id": spec.artifact_id,
        "policy_type": model_config["type"],
        "unique_episodes": spec.unique_episodes,
        "optimizer_steps": training_step,
        "batch_size": int(train_config["batch_size"]),
        "use_amp": str(bool(train_config["policy"]["use_amp"])).lower(),
        "checkpoint_path": checkpoint_path,
        "model_size_bytes": required["model"].stat().st_size,
        "model_sha256": sha256_file(required["model"]),
        "train_config_sha256": sha256_file(required["train_config"]),
        "hub_repo_id": "",
        "publication_status": "local_only",
    }


def write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "model_artifacts.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    rows = [collect_artifact(repo_root, spec) for spec in ARTIFACTS]
    write_inventory(args.output, rows)
    for row in rows:
        print(
            f"{row['artifact_id']}: {row['model_size_bytes']} bytes "
            f"sha256={row['model_sha256']}"
        )
    print(f"wrote {len(rows)} artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
