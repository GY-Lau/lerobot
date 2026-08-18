#!/usr/bin/env python3
"""Measure red/yellow cube locations without adding markers to the scene."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x_px: float
    y_px: float
    u: float
    v: float
    area_px: int
    bbox: tuple[int, int, int, int]
    ambiguous: bool


def has_implausible_size(detection: Detection, image_shape: tuple[int, ...]) -> bool:
    """Reject color regions that are too large to be one cube in this image.

    Use resolution-relative limits because a valid cube near the camera can be
    wider than 200 pixels in a 640x480 image.
    """
    image_height, image_width = image_shape[:2]
    _, _, bbox_width, bbox_height = detection.bbox
    return (
        bbox_width > image_width * 0.5
        or bbox_height > image_height * 0.5
        or detection.area_px > image_width * image_height * 0.15
    )


def _color_mask(image_rgb: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    if color == "red":
        mask = (
            ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
            & (hsv[:, :, 1] >= 100)
            & (hsv[:, :, 2] >= 70)
        )
    elif color == "yellow":
        mask = (
            (hsv[:, :, 0] >= 15)
            & (hsv[:, :, 0] <= 42)
            & (hsv[:, :, 1] >= 55)
            & (hsv[:, :, 2] >= 80)
        )
    else:
        raise ValueError(f"Unsupported color: {color}")

    binary = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def detect_cube(
    image_rgb: np.ndarray,
    color: str,
    min_area_px: int = 500,
    ignore_clipped_rivals: bool = False,
) -> Detection:
    """Locate the largest blob of `color`.

    `ambiguous` warns that a second blob is close enough in size that the wrong
    one may have been picked. Fixtures at the edge of frame -- a wall socket, a
    table edge -- can trip that even though a partially visible blob can never be
    the cube: the caller already requires the cube to be fully in frame. Pass
    `ignore_clipped_rivals=True` to leave border-touching blobs out of that
    comparison. It defaults to False so dataset audits stay reproducible.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got {image_rgb.shape}")

    count, _, stats, centroids = cv2.connectedComponentsWithStats(_color_mask(image_rgb, color))
    components = sorted(
        (
            (int(stats[index, cv2.CC_STAT_AREA]), index)
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_area_px
        ),
        reverse=True,
    )
    if not components:
        raise ValueError(f"No {color} component of at least {min_area_px} pixels was found")

    area, index = components[0]
    rivals = components[1:]
    if ignore_clipped_rivals:
        height_, width_ = image_rgb.shape[:2]
        def _clipped(i: int) -> bool:
            x, y, w, h = (int(stats[i, k]) for k in range(4))
            return x <= 0 or y <= 0 or x + w >= width_ or y + h >= height_
        rivals = [(a, i) for a, i in rivals if not _clipped(i)]
    second_area = rivals[0][0] if rivals else 0
    x, y = (float(value) for value in centroids[index])
    bbox = tuple(int(value) for value in stats[index, :4])
    height, width = image_rgb.shape[:2]
    return Detection(
        x_px=x,
        y_px=y,
        u=x / width,
        v=y / height,
        area_px=area,
        bbox=bbox,
        ambiguous=second_area >= area * 0.5,
    )


def _load_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _dataset_start_frames(repo_id: str, root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root)
    for episode in dataset.meta.episodes:
        episode_index = int(episode["episode_index"])
        frame_index = int(episode["dataset_from_index"])
        image = dataset[frame_index]["observation.images.front"]
        image_rgb = (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        yield episode_index, image_rgb


def _row(source: str | int, image_rgb: np.ndarray) -> dict[str, object]:
    row: dict[str, object] = {"source": source}
    for color in ("red", "yellow"):
        try:
            detection = detect_cube(image_rgb, color)
        except ValueError as error:
            row.update(
                {
                    f"{color}_found": False,
                    f"{color}_x_px": "",
                    f"{color}_y_px": "",
                    f"{color}_u": "",
                    f"{color}_v": "",
                    f"{color}_area_px": "",
                    f"{color}_bbox": "",
                    f"{color}_ambiguous": "",
                    f"{color}_quality": "missing",
                    f"{color}_error": str(error),
                }
            )
        else:
            quality_flags = []
            if detection.ambiguous:
                quality_flags.append("ambiguous")
            if has_implausible_size(detection, image_rgb.shape):
                quality_flags.append("implausible_size")
            row.update(
                {
                    f"{color}_found": True,
                    f"{color}_x_px": f"{detection.x_px:.2f}",
                    f"{color}_y_px": f"{detection.y_px:.2f}",
                    f"{color}_u": f"{detection.u:.6f}",
                    f"{color}_v": f"{detection.v:.6f}",
                    f"{color}_area_px": detection.area_px,
                    f"{color}_bbox": ":".join(str(value) for value in detection.bbox),
                    f"{color}_ambiguous": detection.ambiguous,
                    f"{color}_quality": "+".join(quality_flags) if quality_flags else "clean",
                    f"{color}_error": "",
                }
            )
    return row


def _print_summary(rows: list[dict[str, object]]) -> None:
    print(f"detections: {len(rows)}")
    for color in ("red", "yellow"):
        detected_rows = [row for row in rows if row[f"{color}_found"]]
        if not detected_rows:
            print(f"{color:6s}: no valid detections")
            continue
        clean_rows = [row for row in detected_rows if row[f"{color}_quality"] == "clean"]
        if not clean_rows:
            print(f"{color:6s}: no clean detections after quality filtering")
            continue
        for axis in ("u", "v"):
            values = np.asarray([float(row[f"{color}_{axis}"]) for row in clean_rows])
            percentiles = np.percentile(values, [0, 10, 50, 90, 100])
            formatted = "  ".join(
                f"{label}={value:.3f}"
                for label, value in zip(("min", "p10", "median", "p90", "max"), percentiles)
            )
            print(f"{color:6s} {axis}: {formatted}")
        print(
            f"{color:6s} detected: {len(detected_rows)}/{len(rows)}, "
            f"clean: {len(clean_rows)}/{len(rows)}"
        )
        flagged = [
            f"{row['source']}({row[f'{color}_quality']})"
            for row in detected_rows
            if row[f"{color}_quality"] != "clean"
        ]
        if flagged:
            print(f"{color:6s} flagged sources: {', '.join(flagged)}")
        failures = [str(row["source"]) for row in rows if not row[f"{color}_found"]]
        if failures:
            print(f"{color:6s} missing sources: {', '.join(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Analyze one RGB image")
    source.add_argument("--dataset-root", type=Path, help="Analyze every dataset episode's first frame")
    parser.add_argument("--repo-id", default="GY-William/lerobot_stack_two_cubes")
    parser.add_argument("--output", type=Path, help="Optional output CSV")
    args = parser.parse_args()

    if args.image is not None:
        rows = [_row(args.image.name, _load_image(args.image))]
    else:
        rows = [_row(index, image) for index, image in _dataset_start_frames(args.repo_id, args.dataset_root)]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote: {args.output}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
