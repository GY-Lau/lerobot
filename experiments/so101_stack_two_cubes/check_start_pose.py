#!/usr/bin/env python

"""Check an SO-101 follower against the experiment's reference start pose.

The reference is the per-joint median of frame zero from the 30 demonstration
episodes in ``GY-William/lerobot_stack_two_cubes``. This utility reads joint
positions and never sends an action command.
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

# Rounded bounds covering approximately the central 80% of demonstration
# start poses. Body joints are in degrees; the gripper uses its normalized
# 0-100 range.
TOLERANCE = {
    "shoulder_pan.pos": 2.5,
    "shoulder_lift.pos": 2.5,
    "elbow_flex.pos": 0.5,
    "wrist_flex.pos": 4.0,
    "wrist_roll.pos": 8.0,
    "gripper.pos": 1.0,
}


def check_pose(observation: dict[str, float]) -> tuple[bool, list[tuple]]:
    rows = []
    passed = True
    for joint, target in REFERENCE.items():
        actual = float(observation[joint])
        error = actual - target
        tolerance = TOLERANCE[joint]
        joint_passed = abs(error) <= tolerance
        passed &= joint_passed
        rows.append((joint, target, actual, error, tolerance, joint_passed))
    return passed, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="lerobot_follower_arm")
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

    try:
        robot.connect(calibrate=False)
        observation = robot.get_observation()
    finally:
        if robot.is_connected:
            robot.disconnect()

    passed, rows = check_pose(observation)
    print(f"{'joint':<22} {'target':>9} {'actual':>9} {'error':>9} {'tol':>7}  status")
    for joint, target, actual, error, tolerance, joint_passed in rows:
        status = "PASS" if joint_passed else "FAIL"
        print(f"{joint:<22} {target:>9.2f} {actual:>9.2f} {error:>+9.2f} {tolerance:>7.2f}  {status}")

    print(f"\noverall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

