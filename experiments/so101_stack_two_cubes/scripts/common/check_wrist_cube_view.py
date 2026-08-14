#!/usr/bin/env python3
"""Gate: does the wrist camera see BOTH cubes, fully inside the frame?

Reuses the project's red/yellow detector (analyze_cube_placements). It passes
only if a red cube and a yellow cube are each detected, NOT clipped by the image
border, plausibly sized, and unambiguous. This catches the failure mode where a
cube sits at the very edge of the wrist view and leaves the frame during the
grasp -- exactly what makes the wrist camera useless at the critical moment.

Use it on a captured still, or as a live pre-record gate:

  # a saved frame (e.g. outputs/captured_images/opencv__dev_video0.png)
  check_wrist_cube_view.py --image outputs/captured_images/opencv__dev_video0.png

  # one live frame from the wrist camera (default index 0)
  check_wrist_cube_view.py --camera-index 0 --save-annotated /tmp/wrist_check.png

Exit code 0 = both cubes complete; 2 = fail (missing / clipped / too large /
ambiguous); other = usage or hardware error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Sibling script: its directory is on sys.path when this file is run directly
# or invoked as a subprocess by the recorder.
from analyze_cube_placements import _load_image, detect_cube, has_implausible_size


def border_touch(bbox: tuple[int, int, int, int], shape: tuple[int, ...], margin: int) -> list[str]:
    """Return which image borders the bounding box touches within `margin` px."""
    x, y, w, h = bbox
    height, width = shape[:2]
    sides = []
    if x <= margin:
        sides.append("left")
    if y <= margin:
        sides.append("top")
    if x + w >= width - margin:
        sides.append("right")
    if y + h >= height - margin:
        sides.append("bottom")
    return sides


def grab_camera_frame(index: int, width: int, height: int, warmup: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame_bgr = None
        for _ in range(max(warmup, 1)):  # let auto-exposure/white-balance settle
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the wrist camera")
    finally:
        cap.release()
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def evaluate(image_rgb: np.ndarray, margin: int, min_area: int) -> tuple[bool, dict]:
    results: dict[str, dict] = {}
    ok_all = True
    for color in ("red", "yellow"):
        entry: dict = {"found": False, "complete": False, "flags": []}
        try:
            det = detect_cube(image_rgb, color, min_area_px=min_area)
        except ValueError as error:
            entry["flags"].append("missing")
            entry["error"] = str(error)
            results[color] = entry
            ok_all = False
            continue

        entry["found"] = True
        entry["bbox"] = det.bbox
        entry["area_px"] = det.area_px
        touched = border_touch(det.bbox, image_rgb.shape, margin)
        if touched:
            entry["flags"].append("clipped:" + "+".join(touched))
        if has_implausible_size(det, image_rgb.shape):
            entry["flags"].append("too_large")
        if det.ambiguous:
            entry["flags"].append("ambiguous")
        entry["complete"] = not entry["flags"]
        ok_all &= entry["complete"]
        results[color] = entry
    return ok_all, results


def save_annotated(image_rgb: np.ndarray, results: dict, path: Path) -> None:
    import cv2

    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    palette = {"red": (0, 0, 255), "yellow": (0, 200, 200)}
    for color, entry in results.items():
        if not entry.get("found"):
            continue
        x, y, w, h = entry["bbox"]
        good = entry["complete"]
        box_color = (0, 255, 0) if good else (0, 0, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, 2)
        label = f"{color}:{'OK' if good else '+'.join(entry['flags'])}"
        cv2.putText(canvas, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette[color], 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", type=Path, help="Check a saved RGB image instead of a live grab")
    src.add_argument("--camera-index", type=int, default=0, help="Wrist camera index for a live grab (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=8, help="Frames to discard before the checked frame")
    parser.add_argument("--margin", type=int, default=6, help="Border margin (px) below which a cube counts as clipped")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum colored component area (px)")
    parser.add_argument("--save-annotated", type=Path, help="Write an annotated image showing the detections")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.image is not None:
        image_rgb = _load_image(args.image)
        source = str(args.image)
    else:
        try:
            image_rgb = grab_camera_frame(args.camera_index, args.width, args.height, args.warmup)
        except RuntimeError as error:
            print(f"wrist-view check ERROR: {error}", file=sys.stderr)
            return 3
        source = f"wrist camera index {args.camera_index}"

    ok_all, results = evaluate(image_rgb, args.margin, args.min_area)

    print(f"wrist-view check on {source}  ({image_rgb.shape[1]}x{image_rgb.shape[0]})")
    for color in ("red", "yellow"):
        entry = results[color]
        if not entry["found"]:
            print(f"  {color:6s}: MISSING ({entry.get('error', 'not detected')})")
            continue
        x, y, w, h = entry["bbox"]
        status = "complete" if entry["complete"] else "INCOMPLETE -> " + ", ".join(entry["flags"])
        print(f"  {color:6s}: bbox=({x},{y},{w}x{h}) area={entry['area_px']}  {status}")

    if args.save_annotated is not None:
        save_annotated(image_rgb, results, args.save_annotated)
        print(f"  annotated image: {args.save_annotated}")

    print(f"overall: {'PASS' if ok_all else 'FAIL'}  (both cubes must be present and fully in frame)")
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
