#!/usr/bin/env python3
"""Record the 20-episode vertical-gripper pilot with the red cube on the LEFT.

Placement precision is the remaining bottleneck: at release the wrist camera is
occluded by the held yellow cube and the fixed front camera by the gripper, so
nothing sees the red base cube well. Placing the red cube to the LEFT of the
yellow cube (laterally offset from the descent path) is meant to keep the red
cube visible to the wrist camera during placement. This records a clean,
consistent dataset for that arrangement so the placement action can be learned.

Isolated dataset (do NOT mix arrangements inside one recording). Same cameras,
vertical start-pose gate, and wrist-view scene gate as the vertical pilot.
"""

import os

from record_act_v2_session import RecordingProtocol, main


_wrist_roll = os.environ.get("VERTICAL_WRIST_ROLL_TARGET")
_pose_args = ("--profile", "vertical")
if _wrist_roll:
    _pose_args = _pose_args + ("--wrist-roll-target", _wrist_roll)

_pre_checks = (
    ()
    if os.environ.get("VERTICAL_SKIP_WRIST_CHECK")
    else (("check_wrist_cube_view.py", "--camera-index", "0"),)
)


REDLEFT_PILOT_PROTOCOL = RecordingProtocol(
    name="ACT vertical red-left pilot",
    repo_id="GY-William/lerobot_stack_two_cubes_vertical_redleft_20ep",
    task="Stack the yellow cube on top of the red cube",
    target_episodes=20,
    instruction=(
        "Place the RED cube to the LEFT of the yellow cube (laterally offset), so "
        "the wrist camera can still see the red cube during placement. Gripper "
        "vertical, both cubes fully visible in BOTH camera views. One clean grasp "
        "of the yellow cube, then a LOW, CENTERED release onto the red cube - press "
        "down close before opening so the cube does not roll off. Keep the red-left "
        "arrangement consistent across all 20 episodes. Review at episodes 2 and 10."
    ),
    pose_check_args=_pose_args,
    camera_devices=(("wrist", 0), ("front", 2)),
    pre_record_checks=_pre_checks,
)


if __name__ == "__main__":
    try:
        raise SystemExit(main(REDLEFT_PILOT_PROTOCOL))
    except (KeyboardInterrupt, EOFError):
        print("\nRecording session stopped. Existing episodes remain on disk.")
        raise SystemExit(130)
