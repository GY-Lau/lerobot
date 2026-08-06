#!/usr/bin/env python3
"""Reject gross start-layout confounding between the two language behaviors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROLE_PAIRS = {
    "moving_cube": ("yellow", "red"),
    "base_cube": ("red", "yellow"),
}


def read_positions(path: Path, sources: list[int]) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows_by_source = {int(row["source"]): row for row in csv.DictReader(stream)}
    if set(rows_by_source) != set(sources):
        missing = sorted(set(sources) - set(rows_by_source))
        extra = sorted(set(rows_by_source) - set(sources))
        raise ValueError(f"Unexpected episode sources in {path}: missing={missing}, extra={extra}")

    features: dict[str, np.ndarray] = {}
    for color in ("red", "yellow"):
        bad = [
            source
            for source in sources
            if rows_by_source[source][f"{color}_found"] != "True"
            or rows_by_source[source][f"{color}_quality"] != "clean"
        ]
        if bad:
            raise ValueError(f"{color} detection is not clean in {path}: episodes {bad}")
        for axis in ("u", "v"):
            values = np.asarray(
                [float(rows_by_source[source][f"{color}_{axis}"]) for source in sources],
                dtype=np.float64,
            )
            if np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"{color}_{axis} in {path} is outside normalized image bounds")
            features[f"{color}_{axis}"] = values
    return features


def interval_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first_low, first_high = np.percentile(first, [10, 90])
    second_low, second_high = np.percentile(second, [10, 90])
    intersection = max(0.0, min(first_high, second_high) - max(first_low, second_low))
    narrower_width = min(first_high - first_low, second_high - second_low)
    if narrower_width <= 1e-12:
        return 1.0 if abs(float(np.median(first)) - float(np.median(second))) <= 1e-12 else 0.0
    return float(intersection / narrower_width)


def validate_position_balance(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    *,
    max_median_shift: float = 0.12,
    min_interval_overlap: float = 0.25,
) -> dict[str, dict[str, float]]:
    if max_median_shift < 0:
        raise ValueError("max_median_shift must be non-negative")
    if not 0 <= min_interval_overlap <= 1:
        raise ValueError("min_interval_overlap must be between 0 and 1")

    metrics: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for role, (first_color, second_color) in ROLE_PAIRS.items():
        for axis in ("u", "v"):
            first_values = first[f"{first_color}_{axis}"]
            second_values = second[f"{second_color}_{axis}"]
            median_shift = abs(float(np.median(first_values)) - float(np.median(second_values)))
            overlap = interval_overlap(first_values, second_values)
            key = f"{role}_{axis}"
            metrics[key] = {
                "median_shift": median_shift,
                "p10_p90_overlap": overlap,
            }
            if median_shift > max_median_shift:
                failures.append(
                    f"{key} median shift {median_shift:.3f} exceeds {max_median_shift:.3f}"
                )
            if overlap < min_interval_overlap:
                failures.append(
                    f"{key} p10-p90 overlap {overlap:.3f} is below {min_interval_overlap:.3f}"
                )
    if failures:
        raise ValueError("Language start-layout balance failed: " + "; ".join(failures))
    return metrics


def validate_files(
    first_positions: Path,
    first_manifest: Path,
    second_positions: Path,
    *,
    max_median_shift: float = 0.12,
    min_interval_overlap: float = 0.25,
) -> dict[str, dict[str, float]]:
    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
    first_sources = manifest.get("subsets", {}).get("30", {}).get("episodes")
    if not isinstance(first_sources, list) or len(first_sources) != 30:
        raise ValueError("ACT v2 manifest must declare exactly 30 language-source episodes")
    first = read_positions(first_positions, first_sources)
    second = read_positions(second_positions, list(range(30)))
    return validate_position_balance(
        first,
        second,
        max_median_shift=max_median_shift,
        min_interval_overlap=min_interval_overlap,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("act_v2_positions", type=Path)
    parser.add_argument("act_v2_manifest", type=Path)
    parser.add_argument("inverse_positions", type=Path)
    parser.add_argument("--max-median-shift", type=float, default=0.12)
    parser.add_argument("--min-interval-overlap", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = validate_files(
        args.act_v2_positions,
        args.act_v2_manifest,
        args.inverse_positions,
        max_median_shift=args.max_median_shift,
        min_interval_overlap=args.min_interval_overlap,
    )
    print("Language start-layout balance: PASS")
    for key, values in metrics.items():
        print(
            f"  {key}: median_shift={values['median_shift']:.3f}, "
            f"p10-p90_overlap={values['p10_p90_overlap']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
