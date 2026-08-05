#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("record_act_v2_data.sh")
REPO_ID = Path("GY-William/lerobot_stack_two_cubes_v2")


class RecordActV2ScriptTest(unittest.TestCase):
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

    def write_count(self, home: Path, count: int) -> None:
        meta = home / REPO_ID / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(
            json.dumps({"total_episodes": count}), encoding="utf-8"
        )

    def test_status_reports_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_script(Path(tmp), "--status")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("0/50 recorded episodes", result.stdout)

    def test_dry_run_resumes_and_reports_next_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.write_count(home, 29)
            result = self.run_script(home, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("progress: 29/50 recorded episodes; next episode: 30", result.stdout)
            self.assertIn("--resume=true", result.stdout)

    def test_refuses_episode_51(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.write_count(home, 50)
            result = self.run_script(home, "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to exceed", result.stderr)


if __name__ == "__main__":
    unittest.main()
