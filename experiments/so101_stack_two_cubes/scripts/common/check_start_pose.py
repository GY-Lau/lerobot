#!/usr/bin/env python

"""Check an SO-101 follower against an experiment start-pose envelope.

The reference is the per-joint median of frame zero from the 30 demonstration
episodes in ``GY-William/lerobot_stack_two_cubes``. The default relaxed profile
covers the complete demonstrated start-pose range; the strict profile covers
approximately its central 80%. The dualcam20 profile is derived from the 20
episodes in ``GY-William/lerobot_stack_two_cubes_dualcam_20ep``. This utility
reads joint positions and never sends an action command.
"""

from __future__ import annotations

import argparse

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


REFERENCE = {
    "shoulder_pan.pos": -6.15,
    "shoulder_lift.pos": -103.25,
    "elbow_flex.pos": 97.23,
    "wrist_flex.pos": 53.23,
    "wrist_roll.pos": 1.01,
    "gripper.pos": 2.25,
}

DUALCAM20_REFERENCE = {
    "shoulder_pan.pos": -6.33,
    "shoulder_lift.pos": -103.87,
    "elbow_flex.pos": 96.75,
    "wrist_flex.pos": 50.73,
    "wrist_roll.pos": -83.82,
    "gripper.pos": 2.44,
}

# Vertical-gripper pilot reference, measured live from the follower on
# 2026-08-12 after standing the gripper up (wrist camera back to a side view of
# the cubes). Re-measure and update if the gripper is remounted or recalibrated.
VERTICAL_REFERENCE = {
    "shoulder_pan.pos": -4.40,
    "shoulder_lift.pos": -104.04,
    "elbow_flex.pos": 97.10,
    "wrist_flex.pos": 48.88,
    "wrist_roll.pos": 5.41,
    "gripper.pos": 2.70,
}

# Body joints are in degrees; the gripper uses its normalized 0-100 range.
TOLERANCE_PROFILES = {
    # Rounded symmetric tolerances covering approximately the central 80% of
    # the 30 demonstration start poses.
    "strict": {
        "shoulder_pan.pos": 2.5,
        "shoulder_lift.pos": 2.5,
        "elbow_flex.pos": 0.5,
        "wrist_flex.pos": 4.0,
        "wrist_roll.pos": 8.0,
        "gripper.pos": 1.0,
    },
    # Rounded symmetric tolerances covering the complete observed start-pose
    # range. This is the default evaluation gate: it avoids demanding one
    # exact arm pose while still rejecting starts outside the training support.
    "relaxed": {
        "shoulder_pan.pos": 5.0,
        "shoulder_lift.pos": 5.0,
        "elbow_flex.pos": 4.25,
        "wrist_flex.pos": 16.0,
        "wrist_roll.pos": 13.5,
        "gripper.pos": 18.0,
    },
    # Practical dual-camera evaluation envelope. Allow manually reset body
    # joints to vary, while keeping the gripper close to the demonstrated
    # 2.4-3.1 start range. A half-open gripper was a confirmed evaluation
    # mismatch and should not pass this profile.
    "dualcam20": {
        "shoulder_pan.pos": 3.0,
        "shoulder_lift.pos": 7.0,
        "elbow_flex.pos": 1.5,
        "wrist_flex.pos": 12.0,
        "wrist_roll.pos": 8.0,
        "gripper.pos": 1.0,
    },
    # Vertical-gripper pilot envelope. Same practical tolerances as dualcam20:
    # allow manually reset body joints to vary while keeping the gripper close to
    # its demonstrated closed-start range so a half-open gripper is rejected.
    "vertical": {
        "shoulder_pan.pos": 3.0,
        "shoulder_lift.pos": 7.0,
        "elbow_flex.pos": 1.5,
        "wrist_flex.pos": 12.0,
        "wrist_roll.pos": 8.0,
        "gripper.pos": 3.0,
    },
}

PROFILE_REFERENCES = {
    "strict": REFERENCE,
    "relaxed": REFERENCE,
    "dualcam20": DUALCAM20_REFERENCE,
    "vertical": VERTICAL_REFERENCE,
}


def check_pose(
    observation: dict[str, float],
    tolerances: dict[str, float],
    reference: dict[str, float] = REFERENCE,
) -> tuple[bool, list[tuple]]:
    rows = []
    passed = True
    for joint, target in reference.items():
        actual = float(observation[joint])
        error = actual - target
        tolerance = tolerances[joint]
        joint_passed = abs(error) <= tolerance
        passed &= joint_passed
        rows.append((joint, target, actual, error, tolerance, joint_passed))
    return passed, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="lerobot_follower_arm")
    parser.add_argument(
        "--profile",
        choices=tuple(TOLERANCE_PROFILES),
        default="relaxed",
        help="relaxed covers all demonstrated starts; strict covers the central 80%%",
    )
    parser.add_argument(
        "--wrist-roll-target",
        type=float,
        help=(
            "override the wrist-roll reference after a deliberate recalibration; "
            "other joint references remain unchanged"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        cameras={},
        disable_torque_on_disconnect=True,
    )
    robot = SO101Follower(config)

    # Connect only the motor bus. SO101Follower.connect() also configures the
    # motors and toggles torque, which is unnecessary for a read-only check.
    # The calibration loaded by SO101Follower.__init__ is still applied by the
    # bus when decoding Present_Position.
    try:
        robot.bus.connect()
        observation = robot.get_observation()
    finally:
        if robot.bus.is_connected:
            # Preserve the torque state: this checker must not alter it.
            robot.bus.disconnect(disable_torque=False)

    reference = dict(PROFILE_REFERENCES[args.profile])
    if args.wrist_roll_target is not None:
        reference["wrist_roll.pos"] = args.wrist_roll_target

    passed, rows = check_pose(
        observation,
        TOLERANCE_PROFILES[args.profile],
        reference,
    )
    print(f"profile: {args.profile}")
    print(f"{'joint':<22} {'target':>9} {'actual':>9} {'error':>9} {'tol':>7}  status")
    for joint, target, actual, error, tolerance, joint_passed in rows:
        status = "PASS" if joint_passed else "FAIL"
        print(f"{joint:<22} {target:>9.2f} {actual:>9.2f} {error:>+9.2f} {tolerance:>7.2f}  {status}")

    print(f"\noverall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
