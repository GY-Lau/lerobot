#!/usr/bin/env python3
"""Validate task balance and episode-to-language-label integrity."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def validate_dataset(
    root: Path,
    *,
    min_tasks: int = 2,
    min_episodes_per_task: int = 1,
    expected_tasks: list[str] | None = None,
) -> dict[str, int]:
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    data_paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    for required in (info_path, tasks_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing dataset file: {required}")
    if not data_paths:
        raise FileNotFoundError(f"No parquet data files found below {root / 'data'}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    tasks_table = pq.read_table(tasks_path, columns=["task_index", "task"])
    task_rows = tasks_table.to_pylist()
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    if len(task_by_index) != len(task_rows):
        raise ValueError("tasks.parquet contains duplicate task_index values")
    if len(set(task_by_index.values())) != len(task_by_index):
        raise ValueError("tasks.parquet contains duplicate task descriptions")

    episode_to_task: dict[int, int] = {}
    for data_path in data_paths:
        table = pq.read_table(data_path, columns=["episode_index", "task_index"])
        for row in table.to_pylist():
            episode_index = int(row["episode_index"])
            task_index = int(row["task_index"])
            previous = episode_to_task.setdefault(episode_index, task_index)
            if previous != task_index:
                raise ValueError(
                    f"Episode {episode_index} contains multiple task indices: {previous} and {task_index}"
                )

    unknown = sorted(set(episode_to_task.values()) - set(task_by_index))
    if unknown:
        raise ValueError(f"Data references unknown task indices: {unknown}")

    counts_by_index = Counter(episode_to_task.values())
    counts = {task_by_index[index]: counts_by_index[index] for index in sorted(task_by_index)}
    if len(counts) < min_tasks:
        raise ValueError(f"Expected at least {min_tasks} tasks, found {len(counts)}")
    sparse = {task: count for task, count in counts.items() if count < min_episodes_per_task}
    if sparse:
        raise ValueError(
            f"Tasks below minimum {min_episodes_per_task} episodes: "
            + ", ".join(f"{task!r}={count}" for task, count in sparse.items())
        )
    if expected_tasks is not None:
        missing = sorted(set(expected_tasks) - set(counts))
        extra = sorted(set(counts) - set(expected_tasks))
        if missing or extra:
            raise ValueError(f"Task vocabulary mismatch: missing={missing}, extra={extra}")

    if int(info["total_tasks"]) != len(counts):
        raise ValueError(
            f"info.json total_tasks={info['total_tasks']} but tasks.parquet contains {len(counts)}"
        )
    if int(info["total_episodes"]) != len(episode_to_task):
        raise ValueError(
            f"info.json total_episodes={info['total_episodes']} but data contains "
            f"{len(episode_to_task)} unique episodes"
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--min-tasks", type=int, default=2)
    parser.add_argument("--min-episodes-per-task", type=int, default=1)
    parser.add_argument("--expected-task", action="append", dest="expected_tasks")
    args = parser.parse_args()
    if args.min_tasks < 1 or args.min_episodes_per_task < 1:
        parser.error("minimum values must be positive")

    counts = validate_dataset(
        args.dataset_root.expanduser(),
        min_tasks=args.min_tasks,
        min_episodes_per_task=args.min_episodes_per_task,
        expected_tasks=args.expected_tasks,
    )
    print(f"language dataset: PASS ({sum(counts.values())} episodes, {len(counts)} tasks)")
    for task, count in counts.items():
        print(f"  {count:3d}  {task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
