import csv
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from analyze_cube_placements import detect_cube
from summarize_latency import load_rows, percentile, summarize

import numpy as np


class CubeDetectionTest(unittest.TestCase):
    def test_detects_synthetic_cubes(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[20:50, 20:60] = (255, 0, 0)
        image[55:85, 120:170] = (255, 255, 0)

        red = detect_cube(image, "red", min_area_px=100)
        yellow = detect_cube(image, "yellow", min_area_px=100)

        self.assertAlmostEqual(red.u, 39.5 / 200)
        self.assertAlmostEqual(red.v, 34.5 / 100)
        self.assertAlmostEqual(yellow.u, 144.5 / 200)
        self.assertAlmostEqual(yellow.v, 69.5 / 100)
        self.assertFalse(red.ambiguous)

    def test_rejects_missing_cube(self):
        with self.assertRaisesRegex(ValueError, "No red component"):
            detect_cube(np.zeros((20, 20, 3), dtype=np.uint8), "red", min_area_px=10)


class LatencySummaryTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([0, 10], 50), 5)

    def test_summary_splits_refresh_and_cached(self):
        rows = []
        for refresh, work in ((True, 40.0), (False, 10.0), (False, 20.0)):
            row = {metric: work for metric in (
                "observation_ms",
                "policy_ms",
                "send_action_ms",
                "command_latency_ms",
                "dataset_write_ms",
                "work_ms",
                "loop_period_ms",
            )}
            row["policy_refresh"] = refresh
            rows.append(row)

        result = summarize(rows, target_fps=30)
        self.assertEqual(result["groups"]["refresh"]["frames"], 1)
        self.assertEqual(result["groups"]["cached"]["frames"], 2)
        self.assertAlmostEqual(result["groups"]["all"]["deadline_miss_rate"], 1 / 3)

    def test_load_rows_excludes_warmup(self):
        fields = [
            "warmup",
            "policy_refresh",
            "observation_ms",
            "policy_ms",
            "send_action_ms",
            "command_latency_ms",
            "dataset_write_ms",
            "work_ms",
            "loop_period_ms",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latency.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for warmup in (True, False):
                    row = {field: 1 for field in fields}
                    row["warmup"] = warmup
                    row["policy_refresh"] = False
                    writer.writerow(row)
            self.assertEqual(len(load_rows(path)), 1)
            self.assertEqual(len(load_rows(path, include_warmup=True)), 2)


if __name__ == "__main__":
    unittest.main()
