#!/usr/bin/env python3
"""Validate the ACT v2 start-frame detection table before subset selection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = ("red", "yellow")


def validate_positions(path: Path, expected_episodes: int = 50) -> dict[str, tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} rows, found {len(rows)}")

    indices = [int(row["source"]) for row in rows]
    expected_indices = list(range(expected_episodes))
    if indices != expected_indices:
        raise ValueError(f"Episode indices must be exactly {expected_indices}, found {indices}")

    ranges: dict[str, tuple[float, float]] = {}
    for color in COLORS:
        bad = [
            int(row["source"])
            for row in rows
            if row[f"{color}_found"] != "True" or row[f"{color}_quality"] != "clean"
        ]
        if bad:
            raise ValueError(f"{color} start-frame detection is not clean in episodes {bad}")
        for axis in ("u", "v"):
            values = [float(row[f"{color}_{axis}"]) for row in rows]
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError(f"{color}_{axis} contains values outside normalized image bounds")
            ranges[f"{color}_{axis}"] = (min(values), max(values))
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positions_csv", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=50)
    args = parser.parse_args()
    if args.expected_episodes < 1:
        parser.error("--expected-episodes must be positive")
    ranges = validate_positions(args.positions_csv, args.expected_episodes)
    print(f"ACT v2 positions: PASS ({args.expected_episodes} clean episodes)")
    for key, (minimum, maximum) in ranges.items():
        print(f"  {key}: {minimum:.3f} .. {maximum:.3f} (span {maximum - minimum:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
