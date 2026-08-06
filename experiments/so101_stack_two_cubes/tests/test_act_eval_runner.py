#!/usr/bin/env python3

import subprocess
import unittest
from pathlib import Path

from _bootstrap import SCRIPTS_DIR

SCRIPT = SCRIPTS_DIR / "run_act_data_efficiency_trial.sh"
V2_SCRIPT = SCRIPTS_DIR / "run_act_v2_trial.sh"


class ActEvalRunnerTest(unittest.TestCase):
    def run_dry(self, episode_count: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", str(episode_count), "false"],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_each_subset_maps_to_its_checkpoint_and_run_id(self):
        expected_runs = {
            10: "act_stack_two_cubes_10ep_30k",
            20: "act_stack_two_cubes_20ep_30k",
            30: "act_stack_two_cubes_30k",
        }
        for count, train_run in expected_runs.items():
            with self.subTest(count=count):
                result = self.run_dry(count)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"eval_act_{count}ep_30k_protocol_v1", result.stdout)
                self.assertIn(f"outputs/train/{train_run}/checkpoints/030000", result.stdout)

    def test_unknown_episode_count_is_rejected(self):
        result = self.run_dry(15)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be 10, 20, or 30", result.stderr)

    def test_v2_subsets_map_to_distinct_verified_checkpoints(self):
        for count in (30, 50):
            with self.subTest(count=count):
                result = subprocess.run(
                    ["bash", str(V2_SCRIPT), "--dry-run", str(count), "false"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"eval_act_v2_{count}ep_30k_protocol_v1", result.stdout)
                self.assertIn(
                    f"outputs/train/act_stack_two_cubes_v2_{count}ep_30k/checkpoints/030000",
                    result.stdout,
                )
                self.assertIn("act_v2_subsets.json", result.stdout)

    def test_v2_runner_rejects_v1_subset_size(self):
        result = subprocess.run(
            ["bash", str(V2_SCRIPT), "--dry-run", "20"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be 30 or 50", result.stderr)


if __name__ == "__main__":
    unittest.main()
