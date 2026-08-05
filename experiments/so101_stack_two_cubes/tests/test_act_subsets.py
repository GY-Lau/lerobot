#!/usr/bin/env python3

import csv
import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from generate_act_subsets import FEATURE_KEYS, build_manifest, main


class ActSubsetTest(unittest.TestCase):
    def make_positions(self, path: Path, count: int = 6) -> None:
        fieldnames = ["source", *FEATURE_KEYS, "red_quality", "yellow_quality"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(count):
                writer.writerow(
                    {
                        "source": index,
                        "red_u": index / count,
                        "red_v": (count - index) / count,
                        "yellow_u": (index % 3) / 3,
                        "yellow_v": (index % 2) / 2,
                        "red_quality": "clean" if index < 4 else "ambiguous",
                        "yellow_quality": "clean",
                    }
                )

    def test_subsets_are_nested_and_preserve_quality_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "positions.csv"
            self.make_positions(path)
            manifest = build_manifest(path, [3, 6])
            small = manifest["subsets"]["3"]
            full = manifest["subsets"]["6"]
            self.assertTrue(set(small["episodes"]).issubset(full["episodes"]))
            self.assertEqual((small["clean_episodes"], small["flagged_episodes"]), (2, 1))
            self.assertEqual(full["episodes"], list(range(6)))

    def test_manifest_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "positions.csv"
            self.make_positions(path)
            self.assertEqual(build_manifest(path, [3, 6]), build_manifest(path, [3, 6]))

    def test_manifest_records_selected_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "positions.csv"
            self.make_positions(path)
            manifest = build_manifest(path, [3, 6], "example/act_v2")
            self.assertEqual(manifest["dataset_repo_id"], "example/act_v2")

    def test_largest_subset_must_include_all_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "positions.csv"
            self.make_positions(path)
            with self.assertRaisesRegex(ValueError, "Largest subset"):
                build_manifest(path, [3])


if __name__ == "__main__":
    unittest.main()
