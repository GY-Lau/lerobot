#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _bootstrap import SCRIPTS_DIR

PREPARE = SCRIPTS_DIR / "prepare_act_v2_subsets.sh"
TRAIN = SCRIPTS_DIR / "train_act_v2.sh"
TRAIN_SEQUENCE = SCRIPTS_DIR / "train_act_v2_sequence.sh"


class ActV2PipelineTest(unittest.TestCase):
    def environment(self, tmp: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["LEROBOT_PYTHON"] = sys.executable
        env["ACT_V2_DATASET_ROOT"] = str(tmp / "dataset")
        env["ACT_V2_POSITIONS_CSV"] = str(tmp / "positions.csv")
        env["ACT_V2_MANIFEST"] = str(tmp / "manifest.json")
        return env

    def test_prepare_dry_run_declares_30_50_nested_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(PREPARE), "--dry-run"],
                check=False,
                capture_output=True,
                text=True,
                env=self.environment(Path(tmp)),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--sizes 30 50", result.stdout)
            self.assertIn("validate_act_v2_positions.py", result.stdout)

    def test_train_dry_run_uses_manifest_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "dataset_repo_id": "GY-William/lerobot_stack_two_cubes_v2",
                "subsets": {"30": {"episodes": list(range(20, 50))}},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                ["bash", str(TRAIN), "--dry-run", "30", "act_v2_test"],
                check=False,
                capture_output=True,
                text=True,
                env=self.environment(root),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(r"--dataset.episodes=\[20", result.stdout)
            self.assertIn("--steps=30000", result.stdout)

    def test_train_rejects_unsupported_subset_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(TRAIN), "--dry-run", "40", "act_v2_test"],
                check=False,
                capture_output=True,
                text=True,
                env=self.environment(Path(tmp)),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must be 30 or 50", result.stderr)

    def test_sequence_dry_run_orders_and_verifies_both_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "dataset_repo_id": "GY-William/lerobot_stack_two_cubes_v2",
                "subsets": {
                    "30": {"episodes": list(range(30))},
                    "50": {"episodes": list(range(50))},
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                ["bash", str(TRAIN_SEQUENCE), "--dry-run", "v2_30", "v2_50"],
                check=False,
                capture_output=True,
                text=True,
                env=self.environment(root),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertLess(result.stdout.index("v2_30"), result.stdout.index("v2_50"))
            self.assertEqual(result.stdout.count("verification command:"), 2)
            self.assertIn("--episode-count 30", result.stdout)
            self.assertIn("--episode-count 50", result.stdout)


if __name__ == "__main__":
    unittest.main()
