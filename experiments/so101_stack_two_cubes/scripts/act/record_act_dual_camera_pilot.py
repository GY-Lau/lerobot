#!/usr/bin/env python3
"""Record the isolated 20-episode dual-camera ACT pilot."""

import sys
from pathlib import Path

# Shared recording protocol lives in ../common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from record_act_v2_session import RecordingProtocol, main


DUAL_CAMERA_PROTOCOL = RecordingProtocol(
    name="ACT dual-camera pilot",
    repo_id="GY-William/lerobot_stack_two_cubes_dualcam_20ep",
    task="Stack the yellow cube on top of the red cube",
    target_episodes=20,
    instruction=(
        "Keep both camera mounts and the black cable fixed. Use one clean attempt: "
        "center the yellow cube in the gripper, then lower and center it over the "
        "red cube before release. Review the retained dataset at episodes 2 and 10."
    ),
    pose_check_args=("--wrist-roll-target", "-79.69"),
    camera_devices=(("wrist", 0), ("front", 2)),
)


if __name__ == "__main__":
    try:
        raise SystemExit(main(DUAL_CAMERA_PROTOCOL))
    except (KeyboardInterrupt, EOFError):
        print("\nRecording session stopped. Existing episodes remain on disk.")
        raise SystemExit(130)
