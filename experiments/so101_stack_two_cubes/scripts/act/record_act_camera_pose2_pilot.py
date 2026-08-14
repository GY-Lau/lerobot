#!/usr/bin/env python3
"""Record the isolated 20-episode pilot for camera pose 2."""

import sys
from pathlib import Path

# Shared recording protocol lives in ../common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from record_act_v2_session import RecordingProtocol, main


CAMERA_POSE2_PROTOCOL = RecordingProtocol(
    name="ACT camera-pose-2 pilot",
    repo_id="GY-William/lerobot_stack_two_cubes_pose2_20ep",
    task="Stack the yellow cube on top of the red cube",
    target_episodes=20,
    instruction=(
        "Keep camera pose 2 rigid and both cubes fully visible. "
        "Use one clean attempt: center the yellow cube in the gripper, "
        "then lower and center it over the red cube before release."
    ),
    # The follower was deliberately recalibrated before this isolated pilot.
    # Its unchanged physical start orientation now reads about -79.69 degrees.
    pose_check_args=("--wrist-roll-target", "-79.69"),
)


if __name__ == "__main__":
    try:
        raise SystemExit(main(CAMERA_POSE2_PROTOCOL))
    except (KeyboardInterrupt, EOFError):
        print("\nRecording session stopped. Existing episodes remain on disk.")
        raise SystemExit(130)
