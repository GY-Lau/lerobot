import tempfile
import unittest
from pathlib import Path

from generate_trial_plan import generate_plan, validate_cells
from log_trial import (
    FIELDNAMES,
    append_trial,
    next_trial_index,
    read_rows,
    validate_outcome,
    validate_trial_has_recorded_episode,
)
from summarize_trials import summarize, wilson_interval


def make_row(run_id: str, trial_index: int, success: bool, failure_label: str = "") -> dict:
    values = {
        "run_id": run_id,
        "trial_index": trial_index,
        "placement_regime": "fixed",
        "yellow_position": "Y0",
        "red_position": "R0",
        "yellow_yaw_deg": 0.0,
        "red_yaw_deg": 0.0,
        "success": str(success).lower(),
        "failure_label": failure_label,
        "completion_time_s": 12.0,
        "video_path": "",
        "notes": "",
    }
    assert list(values) == FIELDNAMES
    return values


class EvaluationToolsTest(unittest.TestCase):
    def test_append_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trials.csv"
            append_trial(path, make_row("act_30k_fixed", 1, True))
            append_trial(path, make_row("act_30k_fixed", 2, False, "grasp_failure"))
            rows = read_rows(path)

            self.assertEqual(next_trial_index(rows, "act_30k_fixed"), 3)
            summary = summarize(rows)["act_30k_fixed"]
            self.assertEqual(summary["trials"], 2)
            self.assertEqual(summary["successes"], 1)
            self.assertEqual(summary["success_rate"], 0.5)
            self.assertEqual(summary["failures"]["grasp_failure"], 1)

    def test_outcome_validation(self) -> None:
        validate_outcome(True, None)
        validate_outcome(False, "timeout")
        with self.assertRaises(ValueError):
            validate_outcome(True, "timeout")
        with self.assertRaises(ValueError):
            validate_outcome(False, None)

    def test_duplicate_trial_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trials.csv"
            row = make_row("run", 1, True)
            append_trial(path, row)
            with self.assertRaisesRegex(ValueError, "Duplicate trial"):
                append_trial(path, row)

    def test_result_requires_corresponding_recorded_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            info_path = dataset_root / "meta" / "info.json"
            info_path.parent.mkdir(parents=True)
            info_path.write_text('{"total_episodes": 2}', encoding="utf-8")

            validate_trial_has_recorded_episode(dataset_root, 2)
            with self.assertRaisesRegex(ValueError, "contains only 2 recorded episode"):
                validate_trial_has_recorded_episode(dataset_root, 3)

    def test_wilson_interval_for_two_of_ten(self) -> None:
        low, high = wilson_interval(2, 10)
        self.assertAlmostEqual(low, 0.0567, places=3)
        self.assertAlmostEqual(high, 0.5098, places=3)

    def test_trial_plan_is_deterministic_and_balanced(self) -> None:
        kwargs = {
            "fixed_yellow": "G8",
            "fixed_red": "M6",
            "yellow_cells": ["F7", "G7", "F8", "G8"],
            "red_cells": ["L5", "M5", "L6", "M6"],
            "yaw_values": [0.0, 45.0, 90.0],
            "seed": 123,
        }
        first = generate_plan(**kwargs)
        second = generate_plan(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 30)
        self.assertEqual(sum(row["placement_regime"] == "fixed" for row in first), 10)
        self.assertEqual(sum(row["placement_regime"] == "bounded_random" for row in first), 10)
        self.assertEqual(sum(row["placement_regime"] == "position_orientation_random" for row in first), 10)

    def test_trial_plan_rejects_overlapping_regions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_cells("G8", "G8", ["G8"], ["G8"])


if __name__ == "__main__":
    unittest.main()
