#!/usr/bin/env python

"""Append one physical evaluation trial to trials.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_TRIALS_PATH = Path(__file__).resolve().parents[1] / "results" / "trials.csv"
FIELDNAMES = [
    "run_id",
    "trial_index",
    "placement_regime",
    "yellow_position",
    "red_position",
    "yellow_yaw_deg",
    "red_yaw_deg",
    "success",
    "failure_label",
    "completion_time_s",
    "video_path",
    "notes",
]
PLACEMENT_REGIMES = ("fixed", "bounded_random", "position_orientation_random")
FAILURE_LABELS = (
    "perception_or_target_selection",
    "approach_miss",
    "grasp_failure",
    "drop_in_transit",
    "placement_miss",
    "unstable_stack",
    "post_success_disturbance",
    "timeout",
    "safety_stop",
    "other",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        return list(reader)


def next_trial_index(rows: list[dict[str, str]], run_id: str) -> int:
    indices = [int(row["trial_index"]) for row in rows if row["run_id"] == run_id]
    return max(indices, default=0) + 1


def validate_outcome(success: bool, failure_label: str | None) -> None:
    if success and failure_label:
        raise ValueError("A successful trial cannot have a failure label.")
    if not success and not failure_label:
        raise ValueError("A failed trial requires --failure-label.")


def validate_trial_has_recorded_episode(dataset_root: Path, trial_index: int) -> None:
    """Reject result rows that do not have a corresponding recorded episode."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info["total_episodes"])
    if trial_index > total_episodes:
        raise ValueError(
            f"Cannot log trial {trial_index}: dataset {dataset_root} contains only "
            f"{total_episodes} recorded episode(s). Correct the existing row instead of appending."
        )


def append_trial(path: Path, row: dict[str, object]) -> None:
    rows = read_rows(path)
    identity = (str(row["run_id"]), str(row["trial_index"]))
    if any((item["run_id"], item["trial_index"]) == identity for item in rows):
        raise ValueError(f"Duplicate trial {identity[0]} #{identity[1]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--placement-regime", choices=PLACEMENT_REGIMES, required=True)
    parser.add_argument("--yellow-position", required=True, help="Marked position or grid cell")
    parser.add_argument("--red-position", required=True, help="Marked position or grid cell")
    parser.add_argument("--yellow-yaw-deg", type=float, default=0.0)
    parser.add_argument("--red-yaw-deg", type=float, default=0.0)
    parser.add_argument("--success", action="store_true")
    parser.add_argument("--failure-label", choices=FAILURE_LABELS)
    parser.add_argument("--completion-time-s", type=float)
    parser.add_argument("--video-path", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--trials-path", type=Path, default=DEFAULT_TRIALS_PATH)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Evaluation dataset root. Defaults to the local GY-William cache for RUN_ID when present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_outcome(args.success, args.failure_label)
    rows = read_rows(args.trials_path)
    trial_index = args.trial_index or next_trial_index(rows, args.run_id)
    if trial_index < 1:
        raise ValueError("Trial index must be positive.")
    dataset_root = args.dataset_root or (
        Path.home() / ".cache" / "huggingface" / "lerobot" / "GY-William" / args.run_id
    )
    validate_trial_has_recorded_episode(dataset_root, trial_index)

    row = {
        "run_id": args.run_id,
        "trial_index": trial_index,
        "placement_regime": args.placement_regime,
        "yellow_position": args.yellow_position,
        "red_position": args.red_position,
        "yellow_yaw_deg": args.yellow_yaw_deg,
        "red_yaw_deg": args.red_yaw_deg,
        "success": str(args.success).lower(),
        "failure_label": args.failure_label or "",
        "completion_time_s": "" if args.completion_time_s is None else args.completion_time_s,
        "video_path": args.video_path,
        "notes": args.notes,
    }
    append_trial(args.trials_path, row)
    print(f"recorded {args.run_id} trial {trial_index}: {'success' if args.success else args.failure_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
