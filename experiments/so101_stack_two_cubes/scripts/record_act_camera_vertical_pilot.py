#!/usr/bin/env python3
"""Record the isolated 20-episode vertical-gripper ACT pilot.

This pilot restores the *vertical* gripper orientation so the wrist camera sees
the cubes from the side (a real grasp-height / depth cue), while keeping the
fixed front camera. Everything else matches the dual-camera pilot, so the only
changed variable versus ``GY-William/lerobot_stack_two_cubes_dualcam_20ep`` is
the gripper orientation / wrist-camera viewpoint. That makes the two 20-episode
datasets directly comparable.

Start-pose gate
---------------
Gates on the ``vertical`` profile in check_start_pose.py, whose reference was
measured live from the follower's vertical start pose. If you remount or
recalibrate the gripper, re-measure and update VERTICAL_REFERENCE there, or pass
a one-off wrist-roll override without editing code:

    export VERTICAL_WRIST_ROLL_TARGET=<wrist_roll your vertical start pose reads>
"""

import os

from record_act_v2_session import RecordingProtocol, main


_wrist_roll = os.environ.get("VERTICAL_WRIST_ROLL_TARGET")
_pose_args = ("--profile", "vertical")
if _wrist_roll:
    _pose_args = _pose_args + ("--wrist-roll-target", _wrist_roll)

# Gate every episode on the wrist camera seeing BOTH cubes fully in frame (no
# clipped cube at the edge). Set VERTICAL_SKIP_WRIST_CHECK=1 to disable.
_pre_checks = (
    ()
    if os.environ.get("VERTICAL_SKIP_WRIST_CHECK")
    else (("check_wrist_cube_view.py", "--camera-index", "0"),)
)


VERTICAL_PILOT_PROTOCOL = RecordingProtocol(
    name="ACT vertical-gripper pilot",
    repo_id="GY-William/lerobot_stack_two_cubes_vertical_20ep",
    task="Stack the yellow cube on top of the red cube",
    target_episodes=20,
    instruction=(
        "Gripper VERTICAL: the wrist camera must see the cube from the side (its "
        "height/side faces), not just the top. Keep BOTH cubes fully visible and "
        "roughly centered in BOTH camera views. One clean attempt: center the "
        "yellow cube in the gripper, then lower and center it over the red cube "
        "before release. Keep the black cable out of both views. Review the "
        "retained videos at episodes 2 and 10 before continuing."
    ),
    pose_check_args=_pose_args,
    # Same physical cameras as the dual-camera pilot: only the gripper orientation
    # changes. wrist=/dev/video0, front=/dev/video2.
    camera_devices=(("wrist", 0), ("front", 2)),
    pre_record_checks=_pre_checks,
)


if __name__ == "__main__":
    try:
        raise SystemExit(main(VERTICAL_PILOT_PROTOCOL))
    except (KeyboardInterrupt, EOFError):
        print("\nRecording session stopped. Existing episodes remain on disk.")
        raise SystemExit(130)
