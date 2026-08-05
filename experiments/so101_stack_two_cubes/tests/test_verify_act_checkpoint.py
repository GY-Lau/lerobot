#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from verify_act_checkpoint import REQUIRED_MODEL_FILES, verify_checkpoint


class VerifyActCheckpointTest(unittest.TestCase):
    def make_checkpoint(self, root: Path) -> tuple[Path, Path]:
        checkpoint = root / "checkpoints" / "030000"
        model_dir = checkpoint / "pretrained_model"
        state_dir = checkpoint / "training_state"
        model_dir.mkdir(parents=True)
        state_dir.mkdir(parents=True)
        for name in REQUIRED_MODEL_FILES:
            (model_dir / name).write_text("x", encoding="utf-8")
        config = {
            "steps": 30000,
            "batch_size": 2,
            "seed": 1000,
            "num_workers": 0,
            "dataset": {"repo_id": "example/data", "episodes": [0, 2]},
            "policy": {"type": "act", "use_amp": False},
            "optimizer": {"lr": 1e-5},
            "scheduler": None,
        }
        (model_dir / "train_config.json").write_text(json.dumps(config), encoding="utf-8")
        (state_dir / "training_step.json").write_text(json.dumps({"step": 30000}), encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "dataset_repo_id": "example/data",
                    "total_episodes": 2,
                    "subsets": {"2": {"episodes": [0, 2]}},
                }
            ),
            encoding="utf-8",
        )
        return checkpoint, manifest

    def test_valid_checkpoint_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, manifest = self.make_checkpoint(Path(tmp))
            result = verify_checkpoint(checkpoint, manifest_path=manifest, episode_count=2)
            self.assertEqual(result["steps"], 30000)
            self.assertEqual(result["episodes"], [0, 2])

    def test_incomplete_step_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, manifest = self.make_checkpoint(Path(tmp))
            (checkpoint / "training_state" / "training_step.json").write_text(
                json.dumps({"step": 5000}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "training_step"):
                verify_checkpoint(checkpoint, manifest_path=manifest, episode_count=2)

    def test_wrong_episode_membership_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, manifest = self.make_checkpoint(Path(tmp))
            config_path = checkpoint / "pretrained_model" / "train_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["dataset"]["episodes"] = [0, 1]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset_episodes"):
                verify_checkpoint(checkpoint, manifest_path=manifest, episode_count=2)

    def test_null_episode_list_means_complete_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, manifest = self.make_checkpoint(Path(tmp))
            config_path = checkpoint / "pretrained_model" / "train_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["dataset"]["episodes"] = None
            config_path.write_text(json.dumps(config), encoding="utf-8")

            manifest.write_text(
                json.dumps(
                    {
                        "dataset_repo_id": "example/data",
                        "total_episodes": 2,
                        "subsets": {"2": {"episodes": [0, 1]}},
                    }
                ),
                encoding="utf-8",
            )

            result = verify_checkpoint(checkpoint, manifest_path=manifest, episode_count=2)
            self.assertEqual(result["episodes"], [0, 1])


if __name__ == "__main__":
    unittest.main()
