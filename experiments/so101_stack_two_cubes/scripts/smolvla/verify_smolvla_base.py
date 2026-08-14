#!/usr/bin/env python3
"""Verify the pinned local SmolVLA base snapshot before PEFT training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "manifests" / "smolvla_base.json"
DEFAULT_ROOT = Path("/home/hai/models/lerobot_smolvla_base_c83c316")
DEFAULT_BACKBONE_ROOT = Path("/home/hai/models/smolvlm2_500m_processor_7b375e1")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file_set(
    root: Path, specification: dict[str, object], label: str
) -> dict[str, str]:
    required_files = specification.get("required_files")
    expected_hashes = specification.get("sha256")
    if not isinstance(required_files, list) or not required_files:
        raise ValueError(f"{label} required_files must be a non-empty list")
    if len(set(required_files)) != len(required_files):
        raise ValueError(f"{label} required_files contains duplicates")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError(f"{label} sha256 must be a non-empty object")
    if not set(expected_hashes).issubset(required_files):
        raise ValueError(f"Every hashed {label} file must also appear in required_files")

    for relative_path in required_files:
        path = root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty pinned {label} file: {path}")

    verified_hashes: dict[str, str] = {}
    for relative_path, expected_hash in expected_hashes.items():
        actual_hash = sha256_file(root / relative_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}"
            )
        verified_hashes[relative_path] = actual_hash
    return verified_hashes


def validate_snapshot(
    root: Path,
    manifest_path: Path,
    backbone_root: Path | None = None,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified_hashes = validate_file_set(root, manifest, "SmolVLA base")
    backbone = manifest.get("backbone")
    backbone_result = None
    if backbone is not None:
        if not isinstance(backbone, dict):
            raise ValueError("Manifest backbone must be an object")
        if backbone_root is None:
            raise ValueError("A backbone_root is required by this manifest")
        backbone_result = {
            "repo_id": backbone["repo_id"],
            "revision": backbone["revision"],
            "required_files": len(backbone["required_files"]),
            "verified_hashes": validate_file_set(
                backbone_root, backbone, "SmolVLM2 processor/config"
            ),
        }
    return {
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "required_files": len(manifest["required_files"]),
        "verified_hashes": verified_hashes,
        "backbone": backbone_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--backbone-root", type=Path, default=DEFAULT_BACKBONE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_snapshot(args.root, args.manifest, args.backbone_root)
    print(
        f"SmolVLA base: PASS ({result['repo_id']}@{result['revision']}, "
        f"{result['required_files']} files)"
    )
    for filename, digest in result["verified_hashes"].items():
        print(f"  {filename}: sha256={digest}")
    if result["backbone"] is not None:
        backbone = result["backbone"]
        print(
            f"SmolVLM2 processor/config: PASS ({backbone['repo_id']}@{backbone['revision']}, "
            f"{backbone['required_files']} files)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
