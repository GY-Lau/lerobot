#!/usr/bin/env python3
"""Release the follower's motor torque so the arm can be moved by hand.

A client that is killed rather than stopped never reaches its disconnect, so
disable_torque_on_disconnect never fires and the arm stays rigid. This connects
and disconnects cleanly, which releases it.
"""
import sys
import lerobot.robots.so_follower  # noqa: F401  registers so101_follower
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
cfg = SO101FollowerConfig(port=port, id="lerobot_follower_arm",
                          disable_torque_on_disconnect=True)
robot = SO101Follower(cfg)
robot.connect(calibrate=False)
print("connected; releasing torque")
robot.disconnect()
print("torque released -- the arm should move freely now")
