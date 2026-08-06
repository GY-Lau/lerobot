#!/usr/bin/env python3
"""Summarize manipulation and instruction-following metrics for SmolVLA trials."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from log_smolvla_language_trial import CONDITIONS, DEFAULT_OUTPUT, read_rows
from summarize_trials import parse_bool, wilson_interval


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"Unknown condition in result file: {row['condition']!r}")
        grouped[row["condition"]].append(row)

    summaries: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        condition_rows = grouped.get(condition, [])
        trials = len(condition_rows)
        manipulation = sum(parse_bool(row["manipulation_success"]) for row in condition_rows)
        instruction = sum(
            parse_bool(row["instruction_following_success"]) for row in condition_rows
        )
        failures = Counter(
            row["failure_label"]
            for row in condition_rows
            if not parse_bool(row["instruction_following_success"])
        )
        summary: dict[str, object] = {
            "trials": trials,
            "manipulation_successes": manipulation,
            "instruction_successes": instruction,
            "failures": failures,
        }
        if trials:
            summary["manipulation_rate"] = manipulation / trials
            summary["instruction_rate"] = instruction / trials
            summary["manipulation_ci95"] = wilson_interval(manipulation, trials)
            summary["instruction_ci95"] = wilson_interval(instruction, trials)
        summaries[condition] = summary
    return summaries


def validate_matrix(summaries: dict[str, dict[str, object]], trials_per_condition: int) -> None:
    mismatches = {
        condition: int(summary["trials"])
        for condition, summary in summaries.items()
        if int(summary["trials"]) != trials_per_condition
    }
    if mismatches:
        details = ", ".join(f"{condition}={count}" for condition, count in mismatches.items())
        raise ValueError(
            f"Language evaluation matrix requires exactly {trials_per_condition} "
            f"trials per condition; found {details}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials-per-condition", type=int, default=10)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def format_rate(successes: int, trials: int, interval: tuple[float, float]) -> str:
    low, high = interval
    return (
        f"{successes}/{trials} ({successes / trials:.1%}, "
        f"Wilson 95% CI {low:.1%}-{high:.1%})"
    )


def main() -> int:
    args = parse_args()
    if args.trials_per_condition < 1:
        raise ValueError("--trials-per-condition must be positive")
    rows = read_rows(args.trials_path)
    summaries = summarize(rows)
    if args.require_complete:
        validate_matrix(summaries, args.trials_per_condition)

    for condition, summary in summaries.items():
        trials = int(summary["trials"])
        print(f"{condition}: {trials}/{args.trials_per_condition} trials")
        if not trials:
            continue
        print(
            "  manipulation: "
            + format_rate(
                int(summary["manipulation_successes"]),
                trials,
                summary["manipulation_ci95"],
            )
        )
        print(
            "  instruction:  "
            + format_rate(
                int(summary["instruction_successes"]),
                trials,
                summary["instruction_ci95"],
            )
        )
        failures = summary["failures"]
        if failures:
            print("  failures: " + ", ".join(f"{key}={value}" for key, value in failures.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
