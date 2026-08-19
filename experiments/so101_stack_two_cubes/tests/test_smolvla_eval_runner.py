#!/usr/bin/env python3

import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from _bootstrap import SCRIPTS_DIR


SCRIPT = SCRIPTS_DIR / "smolvla" / "run_smolvla_language_trial.sh"
TRIALS_CSV = SCRIPTS_DIR.parent / "results" / "smolvla_language_trials.csv"


def unescape(text: str) -> str:
    """Drop the shell quoting printf %q adds, so prompts can be matched whole."""
    return text.replace("\\", "")


class SmolVlaEvalRunnerTest(unittest.TestCase):
    def run_script(self, condition: str):
        return subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", condition],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

    def test_each_condition_sends_its_own_prompt_to_the_policy(self):
        """The prompt is the independent variable, so it must reach the policy.

        Under the synchronous controller this was --dataset.single_task on
        lerobot-record. The matrix now runs asynchronously -- synchronous
        SmolVLA scored 0/5 in this project, which would have floored
        manipulation success in both colour orders and left instruction
        following unmeasurable -- so the prompt travels as the client's --task.
        """
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
                stdout = unescape(result.stdout)
                self.assertIn(f"--task={prompt}", stdout)
                self.assertIn(f"eval_smolvla_{condition}", stdout)
                self.assertIn("--pretrained_name_or_path=", stdout)

    def test_every_condition_runs_asynchronously(self):
        for condition in ("yellow_exact", "red_paraphrase"):
            with self.subTest(condition=condition):
                stdout = unescape(self.run_script(condition).stdout)
                self.assertIn("controller=async", stdout)
                self.assertIn("async_robot_client.py", stdout)
                self.assertNotIn("lerobot-record", stdout)

    def test_horizon_is_pinned_and_identical_across_conditions(self):
        """An unbalanced horizon would confound which order the robot chose."""
        horizons = set()
        for condition in ("yellow_exact", "yellow_paraphrase", "red_exact", "red_paraphrase"):
            stdout = self.run_script(condition).stdout
            line = next(l for l in stdout.splitlines() if l.startswith("horizon"))
            horizons.add(line.split(":", 1)[1].strip())
        self.assertEqual(horizons, {"40s"})

    def test_eleventh_trial_is_refused(self):
        """The 4x10 matrix is balanced by design; the cap is what keeps it so."""
        if TRIALS_CSV.exists():
            self.skipTest("real trial log present; refusing to touch it")
        with tempfile.TemporaryDirectory() as tmp:
            fieldnames = ["condition", "trial_index"]
            TRIALS_CSV.parent.mkdir(parents=True, exist_ok=True)
            with TRIALS_CSV.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(1, 11):
                    writer.writerow({"condition": "yellow_exact", "trial_index": i})
            try:
                refused = self.run_script("yellow_exact")
                allowed = self.run_script("red_exact")
            finally:
                TRIALS_CSV.unlink()
                Path(tmp)  # keep the context manager honest
        self.assertEqual(refused.returncode, 1)
        self.assertIn("refusing trial 11", refused.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertIn("trial 1 of 10", allowed.stdout)

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
