#!/usr/bin/env python3
"""Append one SmolVLA physical language-control result with consistent semantics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results" / "smolvla_language_trials.csv"
CONDITIONS = {
    "yellow_exact": (
        "yellow_on_red",
        "exact",
        "Stack the yellow cube on top of the red cube",
    ),
    "yellow_paraphrase": (
        "yellow_on_red",
        "paraphrase",
        "Put the yellow block on the red block",
    ),
    "red_exact": (
        "red_on_yellow",
        "exact",
        "Stack the red cube on top of the yellow cube",
    ),
    "red_paraphrase": (
        "red_on_yellow",
        "paraphrase",
        "Put the red block on the yellow block",
    ),
}
OBSERVED_BEHAVIORS = ("yellow_on_red", "red_on_yellow", "none")
FAILURE_LABELS = (
    "wrong_color_order",
    "perception_or_target_selection",
    "approach_miss",
    "grasp_failure",
    "drop_in_transit",
    "placement_miss",
    "unstable_stack",
    "timeout",
    "safety_stop",
    "other",
)
FIELDNAMES = [
    "condition",
    "trial_index",
    "intended_behavior",
    "prompt_condition",
    "prompt",
    "observed_behavior",
    "manipulation_success",
    "instruction_following_success",
    "failure_label",
    "completion_time_s",
    "scoring",
    "video_path",
    "notes",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        return list(reader)


def next_trial_index(rows: list[dict[str, str]], condition: str) -> int:
    indices = [int(row["trial_index"]) for row in rows if row["condition"] == condition]
    return max(indices, default=0) + 1


def classify_outcome(
    intended_behavior: str,
    observed_behavior: str,
    failure_label: str | None,
) -> tuple[bool, bool]:
    manipulation_success = observed_behavior != "none"
    instruction_success = observed_behavior == intended_behavior

    if instruction_success and failure_label:
        raise ValueError("A successful instruction-following trial cannot have a failure label.")
    if not instruction_success and not failure_label:
        raise ValueError("An unsuccessful instruction-following trial requires --failure-label.")
    if manipulation_success and not instruction_success and failure_label != "wrong_color_order":
        raise ValueError("A stable opposite-order stack must use --failure-label=wrong_color_order.")
    if not manipulation_success and failure_label == "wrong_color_order":
        raise ValueError("wrong_color_order requires a stable stack in the opposite order.")
    return manipulation_success, instruction_success


def validate_recorded_episode(dataset_root: Path, trial_index: int) -> None:
    """Refuse to log a trial whose episode was never recorded.

    Only meaningful for a synchronous trial, which records a dataset. The
    asynchronous controller the language matrix runs under has no dataset code
    at all, so there is nothing to check and --live-scored says so explicitly
    rather than letting the check be silently skipped.
    """
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info["total_episodes"])
    if trial_index > total_episodes:
        raise ValueError(
            f"Cannot log trial {trial_index}: dataset {dataset_root} contains only "
            f"{total_episodes} recorded episode(s)."
        )


def append_row(path: Path, row: dict[str, object]) -> None:
    rows = read_rows(path)
    identity = (str(row["condition"]), str(row["trial_index"]))
    if any((item["condition"], item["trial_index"]) == identity for item in rows):
        raise ValueError(f"Duplicate language trial {identity[0]} #{identity[1]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--observed-behavior", choices=OBSERVED_BEHAVIORS, required=True)
    parser.add_argument("--failure-label", choices=FAILURE_LABELS)
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--completion-time-s", type=float)
    parser.add_argument("--video-path", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--live-scored",
        action="store_true",
        help=(
            "The trial ran under the asynchronous controller, which records no "
            "episode, and was scored by watching the robot. Skips the "
            "recorded-episode check and records how the trial was scored."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    intended_behavior, prompt_condition, prompt = CONDITIONS[args.condition]
    rows = read_rows(args.output)
    trial_index = args.trial_index or next_trial_index(rows, args.condition)
    if trial_index < 1:
        raise ValueError("Trial index must be positive.")
    manipulation_success, instruction_success = classify_outcome(
        intended_behavior, args.observed_behavior, args.failure_label
    )
    dataset_root = args.dataset_root or (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "GY-William"
        / f"eval_smolvla_{args.condition}"
    )
    if not args.live_scored:
        validate_recorded_episode(dataset_root, trial_index)

    append_row(
        args.output,
        {
            "condition": args.condition,
            "trial_index": trial_index,
            "intended_behavior": intended_behavior,
            "prompt_condition": prompt_condition,
            "prompt": prompt,
            "observed_behavior": args.observed_behavior,
            "manipulation_success": str(manipulation_success).lower(),
            "instruction_following_success": str(instruction_success).lower(),
            "failure_label": args.failure_label or "",
            "completion_time_s": (
                "" if args.completion_time_s is None else args.completion_time_s
            ),
            "scoring": "live" if args.live_scored else "recorded",
            "video_path": args.video_path,
            "notes": args.notes,
        },
    )
    print(
        f"recorded {args.condition} trial {trial_index}: "
        f"manipulation_success={str(manipulation_success).lower()} "
        f"instruction_following_success={str(instruction_success).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
