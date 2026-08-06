#!/usr/bin/env python3
"""Verify the pinned local SmolVLA base snapshot before PEFT training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "smolvla_base.json"
DEFAULT_ROOT = Path("/home/hai/models/lerobot_smolvla_base_c83c316")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(root: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_files = manifest.get("required_files")
    expected_hashes = manifest.get("sha256")
    if not isinstance(required_files, list) or not required_files:
        raise ValueError("Manifest required_files must be a non-empty list")
    if len(set(required_files)) != len(required_files):
        raise ValueError("Manifest required_files contains duplicates")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError("Manifest sha256 must be a non-empty object")
    if not set(expected_hashes).issubset(required_files):
        raise ValueError("Every hashed file must also appear in required_files")

    for relative_path in required_files:
        path = root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty pinned SmolVLA file: {path}")

    verified_hashes: dict[str, str] = {}
    for relative_path, expected_hash in expected_hashes.items():
        actual_hash = sha256_file(root / relative_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}"
            )
        verified_hashes[relative_path] = actual_hash
    return {
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "required_files": len(required_files),
        "verified_hashes": verified_hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_snapshot(args.root, args.manifest)
    print(
        f"SmolVLA base: PASS ({result['repo_id']}@{result['revision']}, "
        f"{result['required_files']} files)"
    )
    for filename, digest in result["verified_hashes"].items():
        print(f"  {filename}: sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
