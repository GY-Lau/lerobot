#!/usr/bin/env python3
"""Launch lerobot's async robot client with the config registries populated.

`lerobot.async_inference.robot_client` imports only `RobotConfig` and
`make_robot_from_config`. Robot and camera types register themselves through
`@RobotConfig.register_subclass` decorators that run when their own modules are
imported, and nothing imports them here, so draccus offers an empty choice list
and rejects every `--robot.type`:

    error: argument --robot.type: invalid choice: 'so101_follower' (choose from )

`lerobot_record.py` avoids this by importing the concrete submodules purely for
their side effects (its imports carry `# noqa: F401` for exactly that reason).
This does the same for the subset this experiment uses, then hands over to the
upstream entry point unchanged, so no vendored code is patched.

Takes the same arguments as `python -m lerobot.async_inference.robot_client`.
"""

# ruff: noqa: F401  -- imported for their registration side effects
import lerobot.robots.so_follower  # registers so100_follower and so101_follower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # registers "opencv"

from lerobot.async_inference.robot_client import async_client

if __name__ == "__main__":
    async_client()
