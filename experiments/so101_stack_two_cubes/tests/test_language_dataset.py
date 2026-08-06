#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from _bootstrap import SCRIPTS_DIR
from validate_language_dataset import validate_dataset


PREPARE = SCRIPTS_DIR / "prepare_language_dataset.sh"
TRAIN = SCRIPTS_DIR / "train_smolvla_peft.sh"


class LanguageDatasetValidationTest(unittest.TestCase):
    def make_dataset(self, root: Path, episode_tasks: list[int]) -> None:
        (root / "meta").mkdir(parents=True)
        data_dir = root / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        (root / "meta" / "info.json").write_text(
            json.dumps({"total_tasks": 2, "total_episodes": len(episode_tasks)}),
            encoding="utf-8",
        )
        pq.write_table(
            pa.table(
                {
                    "task_index": [0, 1],
                    "task": ["Stack yellow on red", "Stack red on yellow"],
                }
            ),
            root / "meta" / "tasks.parquet",
        )
        episode_indices = []
        task_indices = []
        for episode_index, task_index in enumerate(episode_tasks):
            episode_indices.extend([episode_index, episode_index])
            task_indices.extend([task_index, task_index])
        pq.write_table(
            pa.table({"episode_index": episode_indices, "task_index": task_indices}),
            data_dir / "file-000.parquet",
        )

    def test_balanced_two_task_dataset_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root, [0, 0, 1, 1])
            counts = validate_dataset(root, min_tasks=2, min_episodes_per_task=2)
            self.assertEqual(counts, {"Stack yellow on red": 2, "Stack red on yellow": 2})

    def test_single_task_usage_fails_balance_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root, [0, 0, 0, 1])
            with self.assertRaisesRegex(ValueError, "below minimum"):
                validate_dataset(root, min_tasks=2, min_episodes_per_task=2)

    def test_multiple_labels_inside_episode_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root, [0, 1])
            data_path = root / "data" / "chunk-000" / "file-000.parquet"
            pq.write_table(
                pa.table({"episode_index": [0, 0, 1, 1], "task_index": [0, 1, 1, 1]}),
                data_path,
            )
            with self.assertRaisesRegex(ValueError, "multiple task indices"):
                validate_dataset(root)

    def test_prepare_dry_run_splits_curated_v2_30_before_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset_repo_id": "GY-William/lerobot_stack_two_cubes_v2",
                        "subsets": {"30": {"episodes": list(range(20, 50))}},
                    }
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["LEROBOT_PYTHON"] = sys.executable
            env["ACT_V2_MANIFEST"] = str(manifest)
            env["HF_LEROBOT_HOME"] = str(root / "datasets")
            result = subprocess.run(
                ["bash", str(PREPARE), "--dry-run"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("split command:", result.stdout)
            self.assertIn("lerobot_stack_two_cubes_v2", result.stdout)
            self.assertIn(r'--operation.splits=\{\"language30\":\[20', result.stdout)
            self.assertIn("lerobot_stack_two_cubes_v2_language30", result.stdout)
            self.assertIn("lerobot_stack_red_on_yellow", result.stdout)
            self.assertIn("lerobot_stack_two_orders_language_v2", result.stdout)
            self.assertIn("validate_language_position_balance.py", result.stdout)

    def test_prepare_rejects_manifest_for_another_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset_repo_id": "someone/another_dataset",
                        "subsets": {"30": {"episodes": list(range(30))}},
                    }
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["LEROBOT_PYTHON"] = sys.executable
            env["ACT_V2_MANIFEST"] = str(manifest)
            result = subprocess.run(
                ["bash", str(PREPARE), "--dry-run"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match", result.stderr)

    def test_smolvla_launcher_defaults_to_v2_language_dataset(self):
        result = subprocess.run(
            ["bash", str(TRAIN), "--dry-run", "smoke", "20", "1", "16", "20", "bf16"],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GY-William/lerobot_stack_two_orders_language_v2", result.stdout)
        self.assertIn("verify_smolvla_base.py", result.stdout)
        self.assertIn("lerobot_smolvla_base_c83c316", result.stdout)
        self.assertIn("smolvlm2_500m_processor_7b375e1", result.stdout)
        self.assertIn("--policy.load_vlm_weights=false", result.stdout)


if __name__ == "__main__":
    unittest.main()
