import tempfile
import unittest
from pathlib import Path

from log_trial import FIELDNAMES, append_trial, next_trial_index, read_rows, validate_outcome
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

    def test_wilson_interval_for_two_of_ten(self) -> None:
        low, high = wilson_interval(2, 10)
        self.assertAlmostEqual(low, 0.0567, places=3)
        self.assertAlmostEqual(high, 0.5098, places=3)


if __name__ == "__main__":
    unittest.main()
