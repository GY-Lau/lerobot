#!/usr/bin/env python3
"""Summarize per-frame latency CSV files produced by lerobot-record."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    "observation_ms",
    "policy_ms",
    "send_action_ms",
    "command_latency_ms",
    "dataset_write_ms",
    "work_ms",
    "loop_period_ms",
)


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_rows(path: Path, include_warmup: bool = False) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = set(METRICS) | {"warmup", "policy_refresh"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing latency column(s): {', '.join(sorted(missing))}")
        rows = []
        for raw in reader:
            warmup = _as_bool(raw["warmup"])
            if warmup and not include_warmup:
                continue
            row: dict[str, object] = {
                metric: float(raw[metric]) for metric in METRICS
            }
            row["warmup"] = warmup
            row["policy_refresh"] = _as_bool(raw["policy_refresh"])
            rows.append(row)
    if not rows:
        raise ValueError("No latency rows remain after filtering")
    return rows


def summarize(rows: list[dict[str, object]], target_fps: float) -> dict[str, object]:
    deadline_ms = 1000 / target_fps
    result: dict[str, object] = {
        "target_fps": target_fps,
        "deadline_ms": deadline_ms,
        "groups": {},
    }
    groups = {
        "all": rows,
        "refresh": [row for row in rows if row["policy_refresh"]],
        "cached": [row for row in rows if not row["policy_refresh"]],
    }
    for name, group in groups.items():
        if not group:
            continue
        summary: dict[str, object] = {
            "frames": len(group),
            "deadline_miss_rate": sum(float(row["work_ms"]) > deadline_ms for row in group) / len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            summary[metric] = {
                "mean": statistics.fmean(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "max": max(values),
            }
        result["groups"][name] = summary
    return result


def _print_summary(summary: dict[str, object]) -> None:
    print(
        f"target: {summary['target_fps']:.1f} FPS "
        f"({summary['deadline_ms']:.2f} ms deadline)"
    )
    for name, group in summary["groups"].items():
        print(
            f"\n{name}: {group['frames']} frames, "
            f"deadline misses={group['deadline_miss_rate']:.1%}"
        )
        print(f"{'metric':24s} {'mean':>9s} {'p50':>9s} {'p95':>9s} {'max':>9s}")
        for metric in METRICS:
            values = group[metric]
            print(
                f"{metric:24s} {values['mean']:9.2f} {values['p50']:9.2f} "
                f"{values['p95']:9.2f} {values['max']:9.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--target-fps", type=float, default=30)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    if args.target_fps <= 0:
        parser.error("--target-fps must be positive")

    summary = summarize(load_rows(args.csv_path, args.include_warmup), args.target_fps)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_summary(summary)


if __name__ == "__main__":
    main()
