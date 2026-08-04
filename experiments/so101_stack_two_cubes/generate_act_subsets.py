#!/usr/bin/env python3
"""Generate deterministic nested ACT episode subsets from start-frame positions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_KEYS = ("red_u", "red_v", "yellow_u", "yellow_v")


@dataclass(frozen=True)
class Episode:
    index: int
    clean: bool
    features: tuple[float | None, ...]


def _read_episodes(path: Path) -> list[Episode]:
    episodes = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            features = tuple(float(row[key]) if row[key] else None for key in FEATURE_KEYS)
            clean = row["red_quality"] == "clean" and row["yellow_quality"] == "clean"
            episodes.append(Episode(int(row["source"]), clean, features))
    if not episodes:
        raise ValueError("Position CSV contains no episodes")
    indices = [episode.index for episode in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError("Position CSV contains duplicate episode indices")
    return sorted(episodes, key=lambda episode: episode.index)


def _standardized_features(episodes: list[Episode]) -> dict[int, np.ndarray]:
    values = np.asarray(
        [[np.nan if value is None else value for value in episode.features] for episode in episodes],
        dtype=np.float64,
    )
    medians = np.nanmedian(values, axis=0)
    if np.isnan(medians).any():
        raise ValueError("At least one position feature is missing for every episode")
    values = np.where(np.isnan(values), medians, values)
    p10, p90 = np.percentile(values, [10, 90], axis=0)
    scales = np.where(p90 > p10, p90 - p10, 1.0)
    standardized = (values - medians) / scales
    return {episode.index: standardized[row] for row, episode in enumerate(episodes)}


def _farthest_order(indices: list[int], features: dict[int, np.ndarray]) -> list[int]:
    if not indices:
        return []
    center = np.median(np.stack([features[index] for index in indices]), axis=0)
    first = min(indices, key=lambda index: (float(np.linalg.norm(features[index] - center)), index))
    selected = [first]
    remaining = set(indices) - {first}
    while remaining:
        next_index = min(
            remaining,
            key=lambda index: (
                -min(float(np.linalg.norm(features[index] - features[chosen])) for chosen in selected),
                index,
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return selected


def build_manifest(path: Path, sizes: list[int]) -> dict[str, object]:
    episodes = _read_episodes(path)
    total = len(episodes)
    if sorted(set(sizes)) != sizes or any(size < 1 or size > total for size in sizes):
        raise ValueError(f"Subset sizes must be unique, increasing, and within 1..{total}")
    if sizes[-1] != total:
        raise ValueError(f"Largest subset must contain all {total} episodes")

    features = _standardized_features(episodes)
    clean_indices = [episode.index for episode in episodes if episode.clean]
    flagged_indices = [episode.index for episode in episodes if not episode.clean]
    clean_order = _farthest_order(clean_indices, features)
    flagged_order = _farthest_order(flagged_indices, features)

    subsets: dict[str, object] = {}
    previous: set[int] = set()
    for size in sizes:
        clean_count = round(size * len(clean_indices) / total)
        flagged_count = size - clean_count
        chosen = set(clean_order[:clean_count] + flagged_order[:flagged_count])
        if len(chosen) != size:
            raise AssertionError("Internal subset size mismatch")
        if not previous.issubset(chosen):
            raise AssertionError("Generated subsets are not nested")
        previous = chosen
        subsets[str(size)] = {
            "episodes": sorted(chosen),
            "clean_episodes": clean_count,
            "flagged_episodes": flagged_count,
        }

    return {
        "dataset_repo_id": "GY-William/lerobot_stack_two_cubes",
        "source_positions_csv": path.name,
        "source_positions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "algorithm": "quality-stratified nested farthest-point traversal v1",
        "feature_keys": list(FEATURE_KEYS),
        "total_episodes": total,
        "total_clean_episodes": len(clean_indices),
        "total_flagged_episodes": len(flagged_indices),
        "subsets": subsets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positions_csv", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("act_data_subsets.json"))
    parser.add_argument("--check", action="store_true", help="Verify that --output matches recomputed content")
    args = parser.parse_args()

    manifest = build_manifest(args.positions_csv, args.sizes)
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"Subset manifest is stale: {args.output}")
        print(f"subset manifest: PASS ({args.output})")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote nested subsets: {args.output}")
    for size, subset in manifest["subsets"].items():
        print(
            f"  {size}: clean={subset['clean_episodes']} flagged={subset['flagged_episodes']} "
            f"episodes={subset['episodes']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
