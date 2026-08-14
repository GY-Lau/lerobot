#!/usr/bin/env python3

import os
import subprocess
import unittest

from _bootstrap import SCRIPTS_DIR


SCRIPT = SCRIPTS_DIR / "smolvla" / "run_smolvla_language_trial.sh"


class SmolVlaEvalRunnerTest(unittest.TestCase):
    def run_script(self, condition: str):
        return subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", condition],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

    def test_each_condition_maps_to_declared_prompt_and_distinct_dataset(self):
        cases = {
            "yellow_exact": "Stack the yellow cube on top of the red cube",
            "yellow_paraphrase": "Put the yellow block on the red block",
            "red_exact": "Stack the red cube on top of the yellow cube",
            "red_paraphrase": "Put the red block on the yellow block",
        }
        for condition, prompt in cases.items():
            with self.subTest(condition=condition):
                result = self.run_script(condition)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"prompt: {prompt}", result.stdout)
                self.assertIn(f"eval_smolvla_{condition}", result.stdout)
                self.assertIn("--policy.path=", result.stdout)
                self.assertIn("--dataset.episode_time_s=20", result.stdout)

    def test_unknown_condition_is_rejected(self):
        result = self.run_script("yellowish")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown CONDITION", result.stderr)

    def test_checkpoint_must_be_six_digits(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", "yellow_exact", "run", "20000"],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("six digits", result.stderr)


if __name__ == "__main__":
    unittest.main()
