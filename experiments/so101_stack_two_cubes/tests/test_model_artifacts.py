#!/usr/bin/env python3

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from inventory_model_artifacts import ArtifactSpec, collect_artifact, write_inventory


class ModelArtifactInventoryTest(unittest.TestCase):
    def make_artifact(self, root: Path, *, saved_step: int = 30_000) -> ArtifactSpec:
        spec = ArtifactSpec("test_act", "test_run", 7)
        checkpoint = root / "outputs/train/test_run/checkpoints/030000"
        model_dir = checkpoint / "pretrained_model"
        state_dir = checkpoint / "training_state"
        model_dir.mkdir(parents=True)
        state_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"model payload")
        (model_dir / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")
        (model_dir / "train_config.json").write_text(
            json.dumps(
                {
                    "steps": 30_000,
                    "batch_size": 2,
                    "policy": {"use_amp": False},
                }
            ),
            encoding="utf-8",
        )
        (state_dir / "training_step.json").write_text(
            json.dumps({"step": saved_step}), encoding="utf-8"
        )
        return spec

    def test_inventory_records_content_hash_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.make_artifact(root)
            row = collect_artifact(root, spec)
            self.assertEqual(row["policy_type"], "act")
            self.assertEqual(row["unique_episodes"], 7)
            self.assertEqual(row["model_sha256"], hashlib.sha256(b"model payload").hexdigest())
            self.assertEqual(row["publication_status"], "local_only")

            output = root / "inventory.csv"
            write_inventory(output, [row])
            self.assertNotIn(b"\r", output.read_bytes())
            with output.open(newline="", encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]["artifact_id"], "test_act")

    def test_incomplete_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.make_artifact(root, saved_step=5_000)
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
                collect_artifact(root, spec)


if __name__ == "__main__":
    unittest.main()
