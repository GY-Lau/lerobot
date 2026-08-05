#!/usr/bin/env python3

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from summarize_act_screening import build_row, write_rows


def latency_row(refresh: bool, loop_ms: float) -> dict[str, object]:
    row: dict[str, object] = {
        "observation_ms": 2.0,
        "policy_ms": 100.0 if refresh else 10.0,
        "send_action_ms": 1.0,
        "command_latency_ms": 103.0 if refresh else 13.0,
        "dataset_write_ms": 0.5,
        "work_ms": 103.5 if refresh else 13.5,
        "loop_period_ms": loop_ms,
        "warmup": False,
        "policy_refresh": refresh,
    }
    return row


class ActScreeningSummaryTest(unittest.TestCase):
    def test_combines_outcome_and_latency(self):
        trial_summary = {
            "trials": 2,
            "successes": 1,
            "success_rate": 0.5,
            "ci95_low": 0.1,
            "ci95_high": 0.9,
            "failures": Counter({"grasp_failure": 1}),
        }
        row = build_row(
            run_id="eval_act_test",
            train_episodes=10,
            trial_summary=trial_summary,
            latency_rows=[latency_row(True, 105.0), latency_row(False, 33.0)],
            target_fps=30.0,
            expected_trials=2,
        )
        self.assertEqual(row["success_rate"], "0.500000")
        self.assertEqual(row["refresh_frames"], 1)
        self.assertEqual(row["cached_frames"], 1)
        self.assertEqual(row["failure_counts"], '{"grasp_failure":1}')

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.csv"
            write_rows(output, [row])
            self.assertNotIn(b"\r", output.read_bytes())

    def test_rejects_incomplete_screening(self):
        trial_summary = {
            "trials": 1,
            "successes": 0,
            "success_rate": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.8,
            "failures": Counter({"timeout": 1}),
        }
        with self.assertRaisesRegex(ValueError, "expected 5"):
            build_row(
                run_id="eval_act_test",
                train_episodes=10,
                trial_summary=trial_summary,
                latency_rows=[latency_row(True, 105.0), latency_row(False, 33.0)],
                target_fps=30.0,
                expected_trials=5,
            )


if __name__ == "__main__":
    unittest.main()
