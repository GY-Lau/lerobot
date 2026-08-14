#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _bootstrap import SCRIPTS_DIR

import record_act_v2_session as session
import record_act_camera_pose2_pilot as pose2_session
import record_act_dual_camera_pilot as dual_session


SCRIPT = SCRIPTS_DIR / "common" / "record_act_v2_session.py"
REPO_ID = Path("GY-William/lerobot_stack_two_cubes_v2")


class RecordActV2SessionTest(unittest.TestCase):
    def test_dual_camera_pilot_has_isolated_dataset_and_camera_mapping(self):
        protocol = dual_session.DUAL_CAMERA_PROTOCOL

        self.assertEqual(protocol.target_episodes, 20)
        self.assertEqual(
            protocol.repo_id,
            "GY-William/lerobot_stack_two_cubes_dualcam_20ep",
        )
        self.assertEqual(protocol.camera_devices, (("wrist", 0), ("front", 2)))
        self.assertEqual(
            protocol.pose_check_args,
            ("--wrist-roll-target", "-79.69"),
        )

    def test_camera_pose2_pilot_is_isolated_and_has_20_episodes(self):
        protocol = pose2_session.CAMERA_POSE2_PROTOCOL

        self.assertEqual(protocol.target_episodes, 20)
        self.assertEqual(
            protocol.repo_id,
            "GY-William/lerobot_stack_two_cubes_pose2_20ep",
        )
        self.assertNotEqual(protocol.repo_id, session.ACT_V2_PROTOCOL.repo_id)
        self.assertEqual(
            protocol.pose_check_args,
            ("--wrist-roll-target", "-79.69"),
        )
        self.assertEqual(session.ACT_V2_PROTOCOL.pose_check_args, ())
        self.assertEqual(session.ACT_V2_PROTOCOL.camera_devices, (("front", 0),))

    def run_script(self, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["HF_LEROBOT_HOME"] = str(home)
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def write_count(self, home: Path, count: int) -> Path:
        root = home / REPO_ID
        meta = root / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(
            json.dumps({"total_episodes": count}), encoding="utf-8"
        )
        if count > 0:
            (meta / "tasks.parquet").touch()
            episodes = meta / "episodes" / "chunk-000"
            episodes.mkdir(parents=True)
            (episodes / "file-000.parquet").touch()
            data = root / "data" / "chunk-000"
            data.mkdir(parents=True)
            (data / "file-000.parquet").touch()
        return root

    def test_dry_run_is_hardware_free_and_does_not_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = self.write_count(home, 0)
            result = self.run_script(home, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("persistent session: 0/50", result.stdout)
            self.assertIn("would be preserved", result.stdout)
            self.assertTrue(root.is_dir())
            self.assertEqual(list(root.parent.glob(f"{root.name}_incomplete_*")), [])

    def test_status_reports_existing_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.write_count(home, 4)
            result = self.run_script(home, "--status")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("4/50 retained episodes", result.stdout)

    def test_incomplete_nonempty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_count(Path(tmp), 3)
            (root / "meta" / "tasks.parquet").unlink()
            with self.assertRaises(RuntimeError):
                session.validate_resumable_dataset(root, 3)

    def test_review_prompt_rejects_unknown_choice(self):
        answers = iter(["maybe", "d"])
        original_input = __builtins__["input"] if isinstance(__builtins__, dict) else __builtins__.input
        try:
            if isinstance(__builtins__, dict):
                __builtins__["input"] = lambda _prompt: next(answers)
            else:
                __builtins__.input = lambda _prompt: next(answers)
            self.assertEqual(session.prompt_review(), "d")
        finally:
            if isinstance(__builtins__, dict):
                __builtins__["input"] = original_input
            else:
                __builtins__.input = original_input


if __name__ == "__main__":
    unittest.main()
