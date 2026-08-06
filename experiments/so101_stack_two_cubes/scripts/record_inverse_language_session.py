#!/usr/bin/env python3
"""Persistent recorder for the red-on-yellow language-control behavior."""

from record_act_v2_session import RecordingProtocol, main


INVERSE_LANGUAGE_PROTOCOL = RecordingProtocol(
    name="inverse language dataset",
    repo_id="GY-William/lerobot_stack_red_on_yellow",
    task="Stack the red cube on top of the yellow cube",
    target_episodes=30,
    instruction="Use one clean attempt; lower and center the red cube before release.",
)


if __name__ == "__main__":
    try:
        raise SystemExit(main(INVERSE_LANGUAGE_PROTOCOL))
    except (KeyboardInterrupt, EOFError):
        print("\nSession stopped; retained episodes were not modified.")
        raise SystemExit(130)
