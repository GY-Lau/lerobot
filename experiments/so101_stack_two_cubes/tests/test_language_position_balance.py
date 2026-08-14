#!/usr/bin/env python3

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _bootstrap import SCRIPTS_DIR
from validate_language_position_balance import validate_position_balance


PREPARE = SCRIPTS_DIR / "smolvla" / "prepare_inverse_language_audit.sh"


class LanguagePositionBalanceTest(unittest.TestCase):
    def features(self, *, offset: float = 0.0, swap_roles: bool = False):
        grid = np.linspace(0.2, 0.8, 30)
        values = {
            "yellow_u": grid + offset,
            "yellow_v": grid[::-1] + offset,
            "red_u": grid * 0.8 + offset,
            "red_v": grid[::-1] * 0.8 + offset,
        }
        if swap_roles:
            values = {
                "yellow_u": grid * 0.8 + offset,
                "yellow_v": grid[::-1] * 0.8 + offset,
                "red_u": grid + offset,
                "red_v": grid[::-1] + offset,
            }
        return values

    def test_role_aligned_matched_distributions_pass(self):
        metrics = validate_position_balance(self.features(), self.features(swap_roles=True))
        self.assertAlmostEqual(metrics["moving_cube_u"]["median_shift"], 0.0)
        self.assertAlmostEqual(metrics["base_cube_v"]["p10_p90_overlap"], 1.0)

    def test_large_role_aligned_shift_fails(self):
        with self.assertRaisesRegex(ValueError, "median shift"):
            validate_position_balance(
                self.features(),
                self.features(offset=0.3, swap_roles=True),
            )

    def test_disjoint_ranges_fail_overlap_gate(self):
        first = self.features()
        second = self.features(swap_roles=True)
        second["red_u"] = np.linspace(0.0, 0.1, 30)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_position_balance(
                first,
                second,
                max_median_shift=1.0,
            )

    def test_audit_dry_run_declares_all_three_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = os.environ.copy()
            env["LEROBOT_PYTHON"] = sys.executable
            env["INVERSE_DATASET_ROOT"] = str(root / "inverse")
            env["INVERSE_POSITIONS_CSV"] = str(root / "inverse.csv")
            env["ACT_V2_POSITIONS_CSV"] = str(root / "v2.csv")
            env["ACT_V2_MANIFEST"] = str(root / "manifest.json")
            result = subprocess.run(
                ["bash", str(PREPARE), "--dry-run"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("analyze_cube_placements.py", result.stdout)
            self.assertIn("validate_act_v2_positions.py", result.stdout)
            self.assertIn("validate_language_position_balance.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
