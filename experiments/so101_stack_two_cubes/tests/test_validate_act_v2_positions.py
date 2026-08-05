#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from validate_act_v2_positions import validate_positions


FIELDS = [
    "source",
    "red_found",
    "red_u",
    "red_v",
    "red_quality",
    "yellow_found",
    "yellow_u",
    "yellow_v",
    "yellow_quality",
]


class ValidateActV2PositionsTest(unittest.TestCase):
    def write_rows(self, path: Path, count: int, *, bad_episode: int | None = None) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for index in range(count):
                writer.writerow(
                    {
                        "source": index,
                        "red_found": True,
                        "red_u": 0.2 + index * 0.01,
                        "red_v": 0.4,
                        "red_quality": "ambiguous" if index == bad_episode else "clean",
                        "yellow_found": True,
                        "yellow_u": 0.7 - index * 0.01,
                        "yellow_v": 0.5,
                        "yellow_quality": "clean",
                    }
                )

    def test_clean_table_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "positions.csv"
            self.write_rows(path, 4)
            ranges = validate_positions(path, 4)
            self.assertAlmostEqual(ranges["red_u"][1], 0.23)

    def test_flagged_detection_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "positions.csv"
            self.write_rows(path, 4, bad_episode=2)
            with self.assertRaisesRegex(ValueError, r"episodes \[2\]"):
                validate_positions(path, 4)

    def test_wrong_episode_count_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "positions.csv"
            self.write_rows(path, 3)
            with self.assertRaisesRegex(ValueError, "Expected 4 rows"):
                validate_positions(path, 4)


if __name__ == "__main__":
    unittest.main()
