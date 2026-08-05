#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("record_inverse_language_data.sh")
REPO_ID = Path("GY-William/lerobot_stack_red_on_yellow")


class RecordInverseScriptTest(unittest.TestCase):
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
            self.assertIn("0/30 recorded episodes", result.stdout)

    def test_dry_run_reports_next_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.write_count(home, 12)
            result = self.run_script(home, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("progress: 12/30 recorded episodes; next episode: 13", result.stdout)
            self.assertIn("--resume=true", result.stdout)

    def test_refuses_episode_31(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.write_count(home, 30)
            result = self.run_script(home, "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to exceed", result.stderr)


if __name__ == "__main__":
    unittest.main()
