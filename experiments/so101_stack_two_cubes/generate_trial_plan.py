#!/usr/bin/env python

"""Generate a deterministic 30-trial cube-placement plan."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).with_name("trial_plan.csv")
FIELDNAMES = [
    "trial_index",
    "placement_regime",
    "yellow_position",
    "red_position",
    "yellow_yaw_deg",
    "red_yaw_deg",
]
CELL_PATTERN = re.compile(r"^[A-S](?:[1-9]|1[0-3])$")


def validate_cells(
    fixed_yellow: str,
    fixed_red: str,
    yellow_cells: list[str],
    red_cells: list[str],
) -> None:
    all_cells = [fixed_yellow, fixed_red, *yellow_cells, *red_cells]
    invalid = [cell for cell in all_cells if not CELL_PATTERN.fullmatch(cell)]
    if invalid:
        raise ValueError(f"Invalid A3 grid cell(s): {', '.join(invalid)}")
    if fixed_yellow not in yellow_cells or fixed_red not in red_cells:
        raise ValueError("Fixed cells must be included in their corresponding candidate lists.")
    overlap = sorted(set(yellow_cells) & set(red_cells))
    if overlap:
        raise ValueError(f"Yellow and red candidate regions must be disjoint: {', '.join(overlap)}")


def generate_plan(
    *,
    fixed_yellow: str,
    fixed_red: str,
    yellow_cells: list[str],
    red_cells: list[str],
    yaw_values: list[float],
    seed: int,
) -> list[dict[str, object]]:
    validate_cells(fixed_yellow, fixed_red, yellow_cells, red_cells)
    if not yaw_values:
        raise ValueError("At least one yaw value is required.")

    rng = random.Random(seed)
    rows = []
    for trial_index in range(1, 31):
        if trial_index <= 10:
            regime = "fixed"
            yellow_position, red_position = fixed_yellow, fixed_red
            yellow_yaw = red_yaw = 0.0
        elif trial_index <= 20:
            regime = "bounded_random"
            yellow_position = rng.choice(yellow_cells)
            red_position = rng.choice(red_cells)
            yellow_yaw = red_yaw = 0.0
        else:
            regime = "position_orientation_random"
            yellow_position = rng.choice(yellow_cells)
            red_position = rng.choice(red_cells)
            yellow_yaw = rng.choice(yaw_values)
            red_yaw = rng.choice(yaw_values)

        rows.append(
            {
                "trial_index": trial_index,
                "placement_regime": regime,
                "yellow_position": yellow_position,
                "red_position": red_position,
                "yellow_yaw_deg": yellow_yaw,
                "red_yaw_deg": red_yaw,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-yellow", required=True)
    parser.add_argument("--fixed-red", required=True)
    parser.add_argument("--yellow-cells", nargs="+", required=True)
    parser.add_argument("--red-cells", nargs="+", required=True)
    parser.add_argument("--yaw-values", nargs="+", type=float, default=[0.0, 45.0, 90.0, 135.0])
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = generate_plan(
        fixed_yellow=args.fixed_yellow,
        fixed_red=args.fixed_red,
        yellow_cells=args.yellow_cells,
        red_cells=args.red_cells,
        yaw_values=args.yaw_values,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} deterministic trials to {args.output} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

