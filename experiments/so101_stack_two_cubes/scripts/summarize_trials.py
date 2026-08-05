#!/usr/bin/env python

"""Summarize physical evaluation trials with Wilson 95% intervals."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from log_trial import DEFAULT_TRIALS_PATH, FIELDNAMES


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Invalid boolean value: {value!r}")
    return normalized == "true"


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    return center - margin, center + margin


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        return list(reader)


def summarize(rows: list[dict[str, str]]) -> dict[str, dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["run_id"]].append(row)

    summaries = {}
    for run_id, run_rows in sorted(grouped.items()):
        successes = sum(parse_bool(row["success"]) for row in run_rows)
        trials = len(run_rows)
        low, high = wilson_interval(successes, trials)
        failures = Counter(row["failure_label"] for row in run_rows if not parse_bool(row["success"]))
        summaries[run_id] = {
            "trials": trials,
            "successes": successes,
            "success_rate": successes / trials,
            "ci95_low": low,
            "ci95_high": high,
            "failures": failures,
        }
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-path", type=Path, default=DEFAULT_TRIALS_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.trials_path)
    if not rows:
        print("No trials recorded.")
        return 0

    for run_id, summary in summarize(rows).items():
        print(run_id)
        print(
            f"  success: {summary['successes']}/{summary['trials']} "
            f"({summary['success_rate']:.1%}, Wilson 95% CI "
            f"{summary['ci95_low']:.1%}-{summary['ci95_high']:.1%})"
        )
        if summary["failures"]:
            print("  failures:")
            for label, count in summary["failures"].most_common():
                print(f"    {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

