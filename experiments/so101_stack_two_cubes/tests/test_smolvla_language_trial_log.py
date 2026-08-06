#!/usr/bin/env python3

import csv
import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import SCRIPTS_DIR  # noqa: F401
from log_smolvla_language_trial import (
    CONDITIONS,
    append_row,
    classify_outcome,
    validate_recorded_episode,
)
from summarize_smolvla_language_trials import summarize, validate_matrix


class SmolVlaLanguageTrialLogTest(unittest.TestCase):
    def test_correct_order_is_both_manipulation_and_instruction_success(self):
        intended, _, _ = CONDITIONS["yellow_exact"]
        self.assertEqual(
            classify_outcome(intended, "yellow_on_red", None),
            (True, True),
        )

    def test_opposite_stable_stack_is_manipulation_success_only(self):
        intended, _, _ = CONDITIONS["yellow_exact"]
        self.assertEqual(
            classify_outcome(intended, "red_on_yellow", "wrong_color_order"),
            (True, False),
        )

    def test_opposite_stack_requires_wrong_order_label(self):
        intended, _, _ = CONDITIONS["red_paraphrase"]
        with self.assertRaisesRegex(ValueError, "wrong_color_order"):
            classify_outcome(intended, "yellow_on_red", "placement_miss")

    def test_missing_stack_rejects_wrong_order_label(self):
        intended, _, _ = CONDITIONS["red_exact"]
        with self.assertRaisesRegex(ValueError, "requires a stable stack"):
            classify_outcome(intended, "none", "wrong_color_order")

    def test_dataset_episode_gate_rejects_unrecorded_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps({"total_episodes": 2}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "only 2 recorded"):
                validate_recorded_episode(root, 3)

    def test_append_rejects_duplicate_condition_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "language.csv"
            row = {
                "condition": "yellow_exact",
                "trial_index": 1,
                "intended_behavior": "yellow_on_red",
                "prompt_condition": "exact",
                "prompt": "Stack the yellow cube on top of the red cube",
                "observed_behavior": "yellow_on_red",
                "manipulation_success": "true",
                "instruction_following_success": "true",
                "failure_label": "",
                "completion_time_s": "12.5",
                "video_path": "",
                "notes": "",
            }
            append_row(path, row)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                append_row(path, row)
            with path.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_summary_keeps_manipulation_and_instruction_rates_separate(self):
        rows = []
        for condition in CONDITIONS:
            intended, prompt_condition, prompt = CONDITIONS[condition]
            opposite = "red_on_yellow" if intended == "yellow_on_red" else "yellow_on_red"
            rows.extend(
                [
                    {
                        "condition": condition,
                        "trial_index": "1",
                        "intended_behavior": intended,
                        "prompt_condition": prompt_condition,
                        "prompt": prompt,
                        "observed_behavior": intended,
                        "manipulation_success": "true",
                        "instruction_following_success": "true",
                        "failure_label": "",
                        "completion_time_s": "",
                        "video_path": "",
                        "notes": "",
                    },
                    {
                        "condition": condition,
                        "trial_index": "2",
                        "intended_behavior": intended,
                        "prompt_condition": prompt_condition,
                        "prompt": prompt,
                        "observed_behavior": opposite,
                        "manipulation_success": "true",
                        "instruction_following_success": "false",
                        "failure_label": "wrong_color_order",
                        "completion_time_s": "",
                        "video_path": "",
                        "notes": "",
                    },
                ]
            )
        summaries = summarize(rows)
        self.assertEqual(summaries["yellow_exact"]["manipulation_rate"], 1.0)
        self.assertEqual(summaries["yellow_exact"]["instruction_rate"], 0.5)
        validate_matrix(summaries, 2)

    def test_incomplete_matrix_is_rejected(self):
        summaries = summarize([])
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            validate_matrix(summaries, 10)


if __name__ == "__main__":
    unittest.main()
