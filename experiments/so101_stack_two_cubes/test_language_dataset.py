#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from validate_language_dataset import validate_dataset


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


if __name__ == "__main__":
    unittest.main()
