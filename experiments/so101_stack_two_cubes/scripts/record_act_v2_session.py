#!/usr/bin/env python3
"""Record ACT v2 demonstrations without re-importing LeRobot every episode."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


REPO_ID = "GY-William/lerobot_stack_two_cubes_v2"
TASK = "Stack the yellow cube on top of the red cube"
TARGET_EPISODES = 50
SCRIPT_DIR = Path(__file__).resolve().parent


def dataset_root() -> Path:
    base = Path(
        os.environ.get(
            "HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"
        )
    )
    return base / REPO_ID


def read_episode_count(root: Path) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 0
    return int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])


def validate_resumable_dataset(root: Path, count: int) -> None:
    if count == 0:
        return
    required_file = root / "meta" / "tasks.parquet"
    episode_files = list((root / "meta" / "episodes").glob("**/*.parquet"))
    data_files = list((root / "data").glob("**/*.parquet"))
    if not required_file.is_file() or not episode_files or not data_files:
        raise RuntimeError(
            f"ACT v2 metadata is incomplete despite declaring {count} episodes; "
            f"refusing to resume: {root}"
        )


def archive_incomplete_dataset(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root.with_name(f"{root.name}_incomplete_{timestamp}")
    shutil.move(str(root), str(backup))
    return backup


def require_devices() -> None:
    missing = [
        path for path in ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/video0") if not Path(path).exists()
    ]
    if missing:
        raise RuntimeError(f"Required device path(s) do not exist: {', '.join(missing)}")


def run_pose_check() -> bool:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "check_start_pose.py"), "--profile", "relaxed"],
        check=False,
    )
    return result.returncode == 0


def discard_last_episode(expected_count: int) -> None:
    env = os.environ.copy()
    env["LEROBOT_PYTHON"] = sys.executable
    subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "discard_last_act_v2_episode.sh"),
            str(expected_count),
        ],
        check=True,
        env=env,
    )


def load_recorder():
    # Deliberately local: --status and --dry-run stay fast and hardware-free.
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

    def build_config(root: Path, resume: bool):
        camera = OpenCVCameraConfig(
            index_or_path=0,
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
        )
        robot = SOFollowerRobotConfig(
            port="/dev/ttyACM0",
            id="lerobot_follower_arm",
            cameras={"front": camera},
        )
        teleop = SOLeaderTeleopConfig(
            port="/dev/ttyACM1",
            id="lerobot_leader_arm",
        )
        dataset = DatasetRecordConfig(
            repo_id=REPO_ID,
            root=root,
            single_task=TASK,
            num_episodes=1,
            episode_time_s=20,
            reset_time_s=0,
            fps=30,
            push_to_hub=False,
        )
        return RecordConfig(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            display_data=False,
            play_sounds=False,
            resume=resume,
        )

    return record, build_config


def prompt_record(count: int) -> str:
    return input(
        f"\nACT v2: {count}/{TARGET_EPISODES} retained. "
        f"[Enter] record episode {count + 1}, [q] quit: "
    ).strip().lower()


def prompt_review() -> str:
    while True:
        answer = input(
            "Review result: [Enter/k] keep, [d] discard, "
            "[q] keep and quit, [x] discard and quit: "
        ).strip().lower()
        if answer in {"", "k", "d", "q", "x"}:
            return answer
        print("Please enter k, d, q, or x.")


def run_session(root: Path) -> int:
    count = read_episode_count(root)
    validate_resumable_dataset(root, count)
    if count >= TARGET_EPISODES:
        print(f"ACT v2 already contains {count}/{TARGET_EPISODES} episodes.")
        return 0

    if root.is_dir() and count == 0:
        backup = archive_incomplete_dataset(root)
        print(f"Preserved incomplete zero-episode dataset at: {backup}")

    require_devices()
    print("Loading LeRobot once (about 25-30 seconds on Jetson) ...", flush=True)
    record, build_config = load_recorder()
    print("Recorder ready. No robot or camera has been connected yet.")

    while True:
        count = read_episode_count(root)
        if count >= TARGET_EPISODES:
            print(f"ACT v2 collection complete: {count}/{TARGET_EPISODES} retained.")
            return 0

        if prompt_record(count) == "q":
            print(f"Stopped with {count}/{TARGET_EPISODES} retained episodes.")
            return 0

        print("Checking follower start pose ...")
        if not run_pose_check():
            print("Start pose check failed; nothing was recorded.")
            continue

        print(f"Episode {count + 1}/{TARGET_EPISODES}: yellow cube on red cube.")
        print("Use one clean attempt; lower and center the yellow cube before release.")
        recording_error = None
        try:
            record(build_config(root, resume=count > 0))
        except Exception as error:  # Keep the session alive after a recoverable device error.
            recording_error = error
            traceback.print_exc()

        new_count = read_episode_count(root)
        if new_count == count:
            print(f"Episode was not saved{': ' + str(recording_error) if recording_error else '.'}")
            continue
        if new_count != count + 1:
            raise RuntimeError(
                f"Unexpected dataset count change: expected {count + 1}, found {new_count}."
            )
        if recording_error is not None:
            print("The recorder raised an error, but one episode was saved. Review it carefully.")

        decision = prompt_review()
        if decision in {"d", "x"}:
            print("Discarding the last episode; LeRobot's dataset editor may take a while ...")
            discard_last_episode(new_count)
            after_discard = read_episode_count(root)
            if after_discard != count:
                raise RuntimeError(
                    f"Discard verification failed: expected {count}, found {after_discard}."
                )
            print(f"Discarded. ACT v2 remains at {count}/{TARGET_EPISODES}.")
        else:
            print(f"Kept. ACT v2 now contains {new_count}/{TARGET_EPISODES} episodes.")

        if decision in {"q", "x"}:
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="show progress without loading LeRobot")
    mode.add_argument("--dry-run", action="store_true", help="show the session plan without changing data")
    args = parser.parse_args()

    root = dataset_root()
    count = read_episode_count(root)
    validate_resumable_dataset(root, count)
    if args.status:
        suffix = " (incomplete empty directory detected)" if root.is_dir() and count == 0 else ""
        print(f"ACT v2 dataset: {count}/{TARGET_EPISODES} retained episodes{suffix}")
        print(f"root: {root}")
        return 0
    if args.dry_run:
        print(f"ACT v2 persistent session: {count}/{TARGET_EPISODES} retained episodes")
        print(f"root: {root}")
        print(f"next episode: {count + 1 if count < TARGET_EPISODES else 'target reached'}")
        print("first startup: one LeRobot import; later episodes reuse the same Python process")
        if root.is_dir() and count == 0:
            print("the incomplete zero-episode directory would be preserved before recording")
        return 0
    return run_session(root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nSession stopped; retained episodes were not modified.")
        raise SystemExit(130)
