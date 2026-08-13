#!/usr/bin/env python3
"""Record ACT v2 demonstrations without re-importing LeRobot every episode."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RecordingProtocol:
    name: str
    repo_id: str
    task: str
    target_episodes: int
    instruction: str
    pose_check_args: tuple[str, ...] = ()
    camera_devices: tuple[tuple[str, int], ...] = (("front", 0),)
    # Extra pre-record gates, each as (script_filename, *args) run with the same
    # interpreter before recording an episode. A non-zero exit skips recording,
    # exactly like a failed start-pose check. Default: none (backward compatible).
    pre_record_checks: tuple[tuple[str, ...], ...] = ()


ACT_V2_PROTOCOL = RecordingProtocol(
    name="ACT v2",
    repo_id="GY-William/lerobot_stack_two_cubes_v2",
    task="Stack the yellow cube on top of the red cube",
    target_episodes=50,
    instruction="Use one clean attempt; lower and center the yellow cube before release.",
)


def dataset_root(protocol: RecordingProtocol = ACT_V2_PROTOCOL) -> Path:
    base = Path(
        os.environ.get(
            "HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"
        )
    )
    return base / protocol.repo_id


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
            f"Dataset metadata is incomplete despite declaring {count} episodes; "
            f"refusing to resume: {root}"
        )


def archive_incomplete_dataset(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root.with_name(f"{root.name}_incomplete_{timestamp}")
    shutil.move(str(root), str(backup))
    return backup


def require_devices(protocol: RecordingProtocol) -> None:
    required_paths = ["/dev/ttyACM0", "/dev/ttyACM1"]
    required_paths.extend(f"/dev/video{index}" for _, index in protocol.camera_devices)
    missing = [path for path in required_paths if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Required device path(s) do not exist: {', '.join(missing)}")


def run_pose_check(protocol: RecordingProtocol) -> bool:
    command = [sys.executable, str(SCRIPT_DIR / "check_start_pose.py")]
    # Default to the relaxed profile unless the protocol selects its own.
    if "--profile" not in protocol.pose_check_args:
        command += ["--profile", "relaxed"]
    command += list(protocol.pose_check_args)
    result = subprocess.run(
        command,
        check=False,
    )
    return result.returncode == 0


def run_extra_checks(protocol: RecordingProtocol) -> bool:
    """Run each configured pre-record gate; return False on the first failure."""
    for check in protocol.pre_record_checks:
        script, *check_args = check
        command = [sys.executable, str(SCRIPT_DIR / script), *check_args]
        if subprocess.run(command, check=False).returncode != 0:
            return False
    return True


def discard_last_episode(protocol: RecordingProtocol, expected_count: int) -> None:
    env = os.environ.copy()
    env["LEROBOT_PYTHON"] = sys.executable
    subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "discard_last_dataset_episode.sh"),
            protocol.repo_id,
            str(expected_count),
        ],
        check=True,
        env=env,
    )


def load_recorder(protocol: RecordingProtocol):
    # Deliberately local: --status and --dry-run stay fast and hardware-free.
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

    def build_config(root: Path, resume: bool):
        cameras = {
            name: OpenCVCameraConfig(
                index_or_path=index,
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
            )
            for name, index in protocol.camera_devices
        }
        robot = SOFollowerRobotConfig(
            port="/dev/ttyACM0",
            id="lerobot_follower_arm",
            cameras=cameras,
        )
        teleop = SOLeaderTeleopConfig(
            port="/dev/ttyACM1",
            id="lerobot_leader_arm",
        )
        dataset = DatasetRecordConfig(
            repo_id=protocol.repo_id,
            root=root,
            single_task=protocol.task,
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


def prompt_record(protocol: RecordingProtocol, count: int) -> str:
    return input(
        f"\n{protocol.name}: {count}/{protocol.target_episodes} retained. "
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


def run_session(root: Path, protocol: RecordingProtocol) -> int:
    count = read_episode_count(root)
    validate_resumable_dataset(root, count)
    if count >= protocol.target_episodes:
        print(f"{protocol.name} already contains {count}/{protocol.target_episodes} episodes.")
        return 0

    if root.is_dir() and count == 0:
        backup = archive_incomplete_dataset(root)
        print(f"Preserved incomplete zero-episode dataset at: {backup}")

    require_devices(protocol)
    print("Loading LeRobot once (about 25-30 seconds on Jetson) ...", flush=True)
    record, build_config = load_recorder(protocol)
    print("Recorder ready. No robot or camera has been connected yet.")

    while True:
        count = read_episode_count(root)
        if count >= protocol.target_episodes:
            print(
                f"{protocol.name} collection complete: "
                f"{count}/{protocol.target_episodes} retained."
            )
            return 0

        if prompt_record(protocol, count) == "q":
            print(f"Stopped with {count}/{protocol.target_episodes} retained episodes.")
            return 0

        print("Checking follower start pose ...")
        if not run_pose_check(protocol):
            print("Start pose check failed; nothing was recorded.")
            continue

        if not run_extra_checks(protocol):
            print("Pre-record scene check failed; nothing was recorded.")
            continue

        print(f"Episode {count + 1}/{protocol.target_episodes}: {protocol.task}.")
        print(protocol.instruction)
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
            discard_last_episode(protocol, new_count)
            after_discard = read_episode_count(root)
            if after_discard != count:
                raise RuntimeError(
                    f"Discard verification failed: expected {count}, found {after_discard}."
                )
            print(f"Discarded. {protocol.name} remains at {count}/{protocol.target_episodes}.")
        else:
            print(
                f"Kept. {protocol.name} now contains "
                f"{new_count}/{protocol.target_episodes} episodes."
            )

        if decision in {"q", "x"}:
            return 0


def main(protocol: RecordingProtocol = ACT_V2_PROTOCOL) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="show progress without loading LeRobot")
    mode.add_argument("--dry-run", action="store_true", help="show the session plan without changing data")
    args = parser.parse_args()

    root = dataset_root(protocol)
    count = read_episode_count(root)
    validate_resumable_dataset(root, count)
    if args.status:
        suffix = " (incomplete empty directory detected)" if root.is_dir() and count == 0 else ""
        print(
            f"{protocol.name} dataset: {count}/{protocol.target_episodes} "
            f"retained episodes{suffix}"
        )
        print(f"root: {root}")
        return 0
    if args.dry_run:
        print(
            f"{protocol.name} persistent session: "
            f"{count}/{protocol.target_episodes} retained episodes"
        )
        print(f"root: {root}")
        print(
            "cameras: "
            + ", ".join(
                f"{name}=/dev/video{index}" for name, index in protocol.camera_devices
            )
        )
        print(
            "next episode: "
            f"{count + 1 if count < protocol.target_episodes else 'target reached'}"
        )
        print("first startup: one LeRobot import; later episodes reuse the same Python process")
        if root.is_dir() and count == 0:
            print("the incomplete zero-episode directory would be preserved before recording")
        return 0
    return run_session(root, protocol)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nSession stopped; retained episodes were not modified.")
        raise SystemExit(130)
