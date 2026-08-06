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
    def make_snapshot(self, root: Path, payload: bytes = b"weights") -> tuple[Path, Path]:
        files = ["config.json", "model.safetensors", "policy_preprocessor.json"]
        for filename in files:
            (root / filename).write_bytes(payload if filename == "model.safetensors" else b"{}")
        backbone_root = root / "backbone"
        backbone_root.mkdir()
        (backbone_root / "config.json").write_bytes(b"backbone config")
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
                    "backbone": {
                        "repo_id": "backbone/test",
                        "revision": "b" * 40,
                        "required_files": ["config.json"],
                        "sha256": {
                            "config.json": hashlib.sha256(b"backbone config").hexdigest()
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest, backbone_root

    def test_snapshot_with_matching_hash_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, backbone_root = self.make_snapshot(root)
            result = validate_snapshot(root, manifest, backbone_root)
            self.assertEqual(result["repo_id"], "lerobot/test")
            self.assertEqual(result["required_files"], 3)
            self.assertEqual(result["backbone"]["repo_id"], "backbone/test")

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, backbone_root = self.make_snapshot(root)
            (root / "model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_snapshot(root, manifest, backbone_root)

    def test_prepare_dry_run_pins_revision_and_local_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, backbone_root = self.make_snapshot(root)
            env = os.environ.copy()
            env["LEROBOT_PYTHON"] = sys.executable
            env["SMOLVLA_BASE_MANIFEST"] = str(manifest)
            env["SMOLVLA_BASE_MODEL"] = str(root / "snapshot")
            env["SMOLVLA_BACKBONE_MODEL"] = str(backbone_root)
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
            self.assertIn(str(backbone_root), result.stdout)
            self.assertIn("processor/config download command", result.stdout)
            self.assertIn("verify_smolvla_base.py", result.stdout)

    def test_prepare_rejects_invalid_retry_count(self):
        env = os.environ.copy()
        env["SMOLVLA_DOWNLOAD_ATTEMPTS"] = "0"
        result = subprocess.run(
            ["bash", str(PREPARE), "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
