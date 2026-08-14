"""Make experiment scripts importable when tests run directly or via discovery."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
# Scripts are split by policy family; every group must be importable by bare name.
for _group in ("common", "act", "smolvla", "diffusion"):
    sys.path.insert(0, str(SCRIPTS_DIR / _group))
sys.path.insert(0, str(SCRIPTS_DIR))
