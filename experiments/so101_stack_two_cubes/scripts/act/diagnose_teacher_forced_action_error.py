#!/usr/bin/env python3
"""Teacher-forced action-error diagnostic for an ACT checkpoint.

Feeds the *expert* observations from a LeRobot dataset to a trained ACT policy
and compares the policy's predicted actions to the recorded expert actions.
Because the real observation is fed at every timestep (teacher forcing), this
isolates the learned perception->action mapping from closed-loop compounding
error. It answers, quantitatively:

  * per-joint 1-step error (in raw action units, e.g. degrees);
  * how the error grows across the predicted action chunk (horizon curve);
  * where in the episode the error concentrates (per-decile "phase" curve);
  * error around the expert grasp (gripper close) and release (gripper open)
    events.

This does NOT roll out the policy on the robot; it is a pure offline read of
"given the true state, does the policy command the right action". Run it on the
machine that holds the checkpoint + dataset (the Jetson).

Example:
  PYTHONNOUSERSITE=1 /home/hai/miniconda3/envs/lerobot/bin/python \
    experiments/so101_stack_two_cubes/scripts/diagnose_teacher_forced_action_error.py \
    --model outputs/train/act_stack_two_cubes_dualcam_20ep_b8_30k_seed1000/checkpoints/030000/pretrained_model \
    --repo-id GY-William/lerobot_stack_two_cubes_dualcam_20ep
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to a .../pretrained_model checkpoint dir")
    p.add_argument("--repo-id", required=True, help="Expert dataset repo id (observations to feed)")
    p.add_argument("--root", default=None, help="Optional local dataset root override")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stride", type=int, default=2, help="Evaluate every Nth frame (speed vs density)")
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes")
    p.add_argument(
        "--out-dir",
        default="outputs/diagnostics",
        help="Directory for the per-frame CSV and summary JSON",
    )
    return p.parse_args()


def normalization_stats(policy: ACTPolicy, dataset: LeRobotDataset) -> tuple[str, np.ndarray, np.ndarray]:
    """Return (mode, a, b) so that raw = decode(norm) per the policy's action norm.

    MEAN_STD: raw = norm * b + a, with a=mean, b=std.
    MIN_MAX : policy space is [-1, 1]; raw = (norm + 1) / 2 * (max - min) + min,
              returned as a=min, b=(max-min) and handled by caller.
    """
    from lerobot.configs.types import FeatureType

    mode = str(policy.config.normalization_mapping.get(FeatureType.ACTION, "MEAN_STD"))
    mode = mode.split(".")[-1]  # NormalizationMode.MEAN_STD -> MEAN_STD
    stats = dataset.meta.stats["action"]
    if "MIN_MAX" in mode:
        return "MIN_MAX", np.asarray(stats["min"], dtype=np.float64), np.asarray(stats["max"], dtype=np.float64)
    return "MEAN_STD", np.asarray(stats["mean"], dtype=np.float64), np.asarray(stats["std"], dtype=np.float64)


def decode(norm: np.ndarray, mode: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if mode == "MIN_MAX":
        return (norm + 1.0) / 2.0 * (b - a) + a
    return norm * b + a


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading policy: {args.model}")
    policy = ACTPolicy.from_pretrained(args.model)
    policy.to(device)
    policy.eval()

    preprocessor, _post = make_pre_post_processors(policy.config, pretrained_path=args.model)

    print(f"Loading expert dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id, root=args.root)

    image_keys = list(policy.config.image_features)
    state_key = "observation.state"
    action_names = dataset.meta.features["action"]["names"]
    if isinstance(action_names, dict):  # some datasets nest names
        action_names = list(action_names.get("motors", action_names))
    action_dim = len(action_names)
    gripper_idx = next((i for i, n in enumerate(action_names) if "gripper" in n.lower()), action_dim - 1)
    chunk = policy.config.chunk_size

    mode, a, b = normalization_stats(policy, dataset)
    print(f"cameras={image_keys}  action_names={action_names}  chunk={chunk}  action_norm={mode}")

    # Group expert actions and global indices per episode. Read from the parquet-backed
    # hf_dataset so this pass does NOT decode any video frames (only the frames we actually
    # evaluate, below, get decoded via dataset[...]).
    hf = dataset.hf_dataset
    ep_actions: dict[int, list[np.ndarray]] = {}
    ep_global_idx: dict[int, list[int]] = {}
    for i in range(len(hf)):
        row = hf[i]
        ep = int(row["episode_index"])
        ep_actions.setdefault(ep, []).append(np.asarray(row["action"], dtype=np.float64))
        ep_global_idx.setdefault(ep, []).append(i)
    episodes = sorted(ep_actions)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "teacher_forced_action_error.csv"

    # Accumulators.
    per_joint_abs_1step: list[np.ndarray] = []          # |pred[0]-expert[t]| per joint (raw units)
    horizon_offsets = [0, 1, 2, 5, 10, 20, 50, min(99, chunk - 1)]
    horizon_abs: dict[int, list[float]] = {k: [] for k in horizon_offsets}  # mean-over-joints L1 (raw)
    decile_abs: list[list[float]] = [[] for _ in range(10)]  # mean-over-joints L1 by episode progress
    grasp_window_abs: dict[int, list[float]] = {d: [] for d in range(-5, 6)}   # around gripper-close
    release_window_abs: dict[int, list[float]] = {d: [] for d in range(-5, 6)}  # around gripper-open

    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(
        ["episode_index", "frame_in_episode", "progress"]
        + [f"pred_{n}" for n in action_names]
        + [f"expert_{n}" for n in action_names]
        + [f"abs_err_{n}" for n in action_names]
    )

    n_eval = 0
    for ep in episodes:
        actions = np.stack(ep_actions[ep])  # (T, action_dim), raw units
        gidx = ep_global_idx[ep]
        T = len(gidx)

        # Detect grasp (gripper close) and release (gripper open) events from the expert
        # gripper trajectory: normalize to [0,1] within the episode, threshold-cross.
        g = actions[:, gripper_idx]
        g_span = float(g.max() - g.min())
        g_norm = (g - g.min()) / g_span if g_span > 1e-6 else np.zeros_like(g)
        closed = g_norm < 0.5  # heuristic: below mid-range = more closed (SO-101 gripper: small=closed)
        close_events = [t for t in range(1, T) if closed[t] and not closed[t - 1]]
        open_events = [t for t in range(1, T) if (not closed[t]) and closed[t - 1]]

        for t in range(0, T, args.stride):
            item = dataset[gidx[t]]
            obs = {k: item[k] for k in image_keys + [state_key]}
            if "task" in item:
                obs["task"] = item["task"]
            batch = preprocessor(obs)
            with torch.inference_mode():
                chunk_norm = policy.predict_action_chunk(batch)  # (1, chunk, action_dim), normalized
            chunk_norm = chunk_norm[0].detach().to("cpu", dtype=torch.float64).numpy()
            chunk_raw = decode(chunk_norm, mode, a, b)  # (chunk, action_dim) raw units

            # 1-step per-joint error.
            err1 = np.abs(chunk_raw[0] - actions[t])
            per_joint_abs_1step.append(err1)

            progress = t / max(T - 1, 1)
            decile = min(int(progress * 10), 9)
            decile_abs[decile].append(float(err1.mean()))

            writer.writerow(
                [ep, t, round(progress, 4)]
                + [round(float(x), 4) for x in chunk_raw[0]]
                + [round(float(x), 4) for x in actions[t]]
                + [round(float(x), 4) for x in err1]
            )

            # Horizon curve: pred[k] vs expert[t+k].
            for k in horizon_offsets:
                if t + k < T:
                    horizon_abs[k].append(float(np.abs(chunk_raw[k] - actions[t + k]).mean()))

            # Event-aligned windows (use the same fresh 1-step error at the offset frame).
            for ev_list, acc in ((close_events, grasp_window_abs), (open_events, release_window_abs)):
                for ev in ev_list:
                    d = t - ev
                    if d in acc:
                        acc[d].append(float(err1.mean()))

            n_eval += 1
            if n_eval % 250 == 0:
                print(f"  ...{n_eval} frames evaluated")

    csv_f.close()

    per_joint = np.stack(per_joint_abs_1step)  # (N, action_dim)
    joint_mae = per_joint.mean(axis=0)
    overall_mae = float(per_joint.mean())

    def m(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "model": args.model,
        "repo_id": args.repo_id,
        "action_norm": mode,
        "frames_evaluated": n_eval,
        "episodes": len(episodes),
        "chunk_size": chunk,
        "overall_1step_mae_raw_units": round(overall_mae, 4),
        "per_joint_1step_mae": {n: round(float(v), 4) for n, v in zip(action_names, joint_mae)},
        "horizon_mae": {str(k): round(m(horizon_abs[k]), 4) for k in horizon_offsets},
        "phase_decile_mae": [round(m(decile_abs[d]), 4) for d in range(10)],
        "grasp_aligned_mae": {str(d): round(m(grasp_window_abs[d]), 4) for d in sorted(grasp_window_abs)},
        "release_aligned_mae": {str(d): round(m(release_window_abs[d]), 4) for d in sorted(release_window_abs)},
    }
    summary_path = out_dir / "teacher_forced_action_error_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Console report.
    print("\n==== Teacher-forced action-error diagnostic ====")
    print(f"frames evaluated : {n_eval}  ({len(episodes)} episodes, stride {args.stride})")
    print(f"action units     : raw (per dataset stats, mostly degrees; gripper in its own unit)")
    print(f"overall 1-step MAE (mean over joints): {overall_mae:.3f}")
    print("\nper-joint 1-step MAE:")
    for n, v in zip(action_names, joint_mae):
        print(f"  {n:<16} {v:8.3f}")
    print("\nhorizon MAE (pred[k] vs expert[t+k], mean over joints):")
    for k in horizon_offsets:
        print(f"  k={k:<3} {m(horizon_abs[k]):8.3f}")
    print("\nphase curve — 1-step MAE by episode progress decile (0=start .. 9=end):")
    for d in range(10):
        bar = "#" * int(min(m(decile_abs[d]), 40))
        print(f"  {d*10:>3}-{d*10+10:<3}%  {m(decile_abs[d]):7.3f}  {bar}")
    print("\ngrasp-aligned (expert gripper close at d=0) 1-step MAE:")
    for d in sorted(grasp_window_abs):
        print(f"  d={d:+d}  {m(grasp_window_abs[d]):7.3f}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
