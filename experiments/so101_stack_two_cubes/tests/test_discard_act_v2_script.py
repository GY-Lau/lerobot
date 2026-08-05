#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _bootstrap import SCRIPTS_DIR

SCRIPT = SCRIPTS_DIR / "discard_last_act_v2_episode.sh"
REPO_ID = Path("GY-William/lerobot_stack_two_cubes_v2")


class DiscardActV2ScriptTest(unittest.TestCase):
    def make_dataset(self, home: Path, count: int) -> None:
        meta = home / REPO_ID / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(
            json.dumps({"total_episodes": count}), encoding="utf-8"
        )

    def run_script(self, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["HF_LEROBOT_HOME"] = str(home)
        env["LEROBOT_PYTHON"] = sys.executable
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_requires_matching_episode_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.make_dataset(home, 4)
            result = self.run_script(home, "--dry-run", "3")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dataset contains 4", result.stderr)

    def test_one_episode_dry_run_preserves_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.make_dataset(home, 1)
            result = self.run_script(home, "--dry-run", "1")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("moving the one-episode dataset", result.stdout)
            self.assertTrue((home / REPO_ID / "meta" / "info.json").is_file())

    def test_multiple_episode_dry_run_targets_last_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.make_dataset(home, 7)
            result = self.run_script(home, "--dry-run", "7")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("last episode index 6", result.stdout)
            self.assertIn(r"episode_indices=\[6\]", result.stdout)


if __name__ == "__main__":
    unittest.main()
