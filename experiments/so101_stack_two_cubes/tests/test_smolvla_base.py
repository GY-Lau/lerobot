#!/usr/bin/env python3

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _bootstrap import SCRIPTS_DIR
from verify_smolvla_base import validate_snapshot


PREPARE = SCRIPTS_DIR / "prepare_smolvla_base.sh"


class SmolVlaBaseTest(unittest.TestCase):
    def make_snapshot(self, root: Path, payload: bytes = b"weights") -> Path:
        files = ["config.json", "model.safetensors", "policy_preprocessor.json"]
        for filename in files:
            (root / filename).write_bytes(payload if filename == "model.safetensors" else b"{}")
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "repo_id": "lerobot/test",
                    "revision": "a" * 40,
                    "required_files": files,
                    "sha256": {
                        "model.safetensors": hashlib.sha256(payload).hexdigest(),
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_snapshot_with_matching_hash_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_snapshot(root)
            result = validate_snapshot(root, manifest)
            self.assertEqual(result["repo_id"], "lerobot/test")
            self.assertEqual(result["required_files"], 3)

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_snapshot(root)
            (root / "model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_snapshot(root, manifest)

    def test_prepare_dry_run_pins_revision_and_local_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_snapshot(root)
            env = os.environ.copy()
            env["LEROBOT_PYTHON"] = sys.executable
            env["SMOLVLA_BASE_MANIFEST"] = str(manifest)
            env["SMOLVLA_BASE_MODEL"] = str(root / "snapshot")
            result = subprocess.run(
                ["bash", str(PREPARE), "--dry-run"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--revision=", result.stdout)
            self.assertIn("a" * 40, result.stdout)
            self.assertIn(str(root / "snapshot"), result.stdout)
            self.assertIn("verify_smolvla_base.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
