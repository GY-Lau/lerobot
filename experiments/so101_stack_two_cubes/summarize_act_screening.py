#!/usr/bin/env python3
"""Combine ACT physical screening outcomes and latency into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize_latency import load_rows as load_latency_rows
from summarize_latency import summarize as summarize_latency
from summarize_trials import read_rows as read_trial_rows
from summarize_trials import summarize as summarize_trials


RUNS = {
    10: "eval_act_10ep_30k_protocol_v1",
    20: "eval_act_20ep_30k_protocol_v1",
    30: "eval_act_30ep_30k_protocol_v1",
}
FIELDNAMES = (
    "run_id",
    "train_episodes",
    "trials",
    "successes",
    "success_rate",
    "ci95_low",
    "ci95_high",
    "failure_counts",
    "latency_frames",
    "refresh_frames",
    "cached_frames",
    "command_p50_ms",
    "command_p95_ms",
    "refresh_policy_p50_ms",
    "refresh_policy_p95_ms",
    "cached_policy_p50_ms",
    "effective_fps",
    "deadline_miss_rate",
    "screening_status",
)


def build_row(
    *,
    run_id: str,
    train_episodes: int,
    trial_summary: dict[str, object],
    latency_rows: list[dict[str, object]],
    target_fps: float,
    expected_trials: int,
) -> dict[str, object]:
    if int(trial_summary["trials"]) != expected_trials:
        raise ValueError(
            f"{run_id} has {trial_summary['trials']} result rows; expected {expected_trials}"
        )
    latency = summarize_latency(latency_rows, target_fps)
    groups = latency["groups"]
    all_frames = groups["all"]
    refresh = groups["refresh"]
    cached = groups["cached"]
    effective_fps = 1000 * len(latency_rows) / sum(float(row["loop_period_ms"]) for row in latency_rows)
    return {
        "run_id": run_id,
        "train_episodes": train_episodes,
        "trials": trial_summary["trials"],
        "successes": trial_summary["successes"],
        "success_rate": f"{float(trial_summary['success_rate']):.6f}",
        "ci95_low": f"{float(trial_summary['ci95_low']):.6f}",
        "ci95_high": f"{float(trial_summary['ci95_high']):.6f}",
        "failure_counts": json.dumps(dict(trial_summary["failures"]), sort_keys=True, separators=(",", ":")),
        "latency_frames": all_frames["frames"],
        "refresh_frames": refresh["frames"],
        "cached_frames": cached["frames"],
        "command_p50_ms": f"{all_frames['command_latency_ms']['p50']:.3f}",
        "command_p95_ms": f"{all_frames['command_latency_ms']['p95']:.3f}",
        "refresh_policy_p50_ms": f"{refresh['policy_ms']['p50']:.3f}",
        "refresh_policy_p95_ms": f"{refresh['policy_ms']['p95']:.3f}",
        "cached_policy_p50_ms": f"{cached['policy_ms']['p50']:.3f}",
        "effective_fps": f"{effective_fps:.3f}",
        "deadline_miss_rate": f"{float(all_frames['deadline_miss_rate']):.6f}",
        "screening_status": "exploratory_n5_not_reportable",
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-path", type=Path, default=here / "trials.csv")
    parser.add_argument("--latency-dir", type=Path, default=here.parents[1] / "outputs" / "eval_latency")
    parser.add_argument("--output", type=Path, default=here / "act_physical_screening.csv")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--expected-trials", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trials = summarize_trials(read_trial_rows(args.trials_path))
    rows = []
    for train_episodes, run_id in RUNS.items():
        if run_id not in trials:
            raise ValueError(f"Missing trial results for {run_id}")
        latency_path = args.latency_dir / f"{run_id}.csv"
        rows.append(
            build_row(
                run_id=run_id,
                train_episodes=train_episodes,
                trial_summary=trials[run_id],
                latency_rows=load_latency_rows(latency_path),
                target_fps=args.target_fps,
                expected_trials=args.expected_trials,
            )
        )
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} screening summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
