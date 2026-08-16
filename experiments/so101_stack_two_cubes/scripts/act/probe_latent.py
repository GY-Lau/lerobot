#!/usr/bin/env python3
"""Does ACT's CVAE latent carry the target choice, or was it never learned?

Background. ACT is a CVAE whose style encoder is fed [cls, robot_state,
action_sequence] and *no images*, so the latent z absorbs whatever in the
demonstrated action the robot state does not explain -- including which of
several valid targets the demonstrator picked. At inference the latent is set to
zero (`use_vae` is true, but the sampling branch in `modeling_act.py` is gated on
`self.training`), so the decoder returns the conditional mean of the action
distribution given the observation. Where the observation does not disambiguate
the target, that mean need not be a valid action.

Merging two datasets whose target layouts do not overlap made this project's
policy release short of the cube, exactly between the two modes. Two hypotheses
produce that same signature:

  A. z encoded the mode, and zeroing it at inference throws the choice away.
  B. the decoder never learned to read the target from pixels at all.

They differ in one measurable way. Under A the decoder is *able* to emit the
right action for some latent, so drawing z from the prior should occasionally
land much closer to the expert than z=0 does. Under B no latent helps, because
the information is not in the network.

For frames around each expert release this reports, against the expert action:
z=0 (what deployment does), best-of-K and mean-of-K over K prior samples, and
the per-joint spread across those samples.

  best-of-K much better than z=0, non-trivial spread -> A
  best-of-K about equal to z=0, spread near zero     -> B

The latent is injected with a forward pre-hook on the module that consumes it,
so the policy implementation is untouched.

Example:
  python probe_latent.py \
    --model outputs/train/act_..._combined_40ep_b8_30k_seed1000/checkpoints/030000/pretrained_model \
    --repo-id GY-William/lerobot_stack_two_cubes_vertical_redleft_20ep
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--repo-id", required=True, help="Expert dataset whose observations are fed")
    p.add_argument("--root", default=None, help="Optional local dataset root override")
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=32, help="K latent draws per frame")
    p.add_argument("--release-window", type=int, default=2,
                   help="Frames each side of the expert release to evaluate")
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default=None, help="Optional JSON summary path")
    return p.parse_args()


def normalization_stats(policy: ACTPolicy, dataset: LeRobotDataset):
    from lerobot.configs.types import FeatureType

    mode = str(policy.config.normalization_mapping.get(FeatureType.ACTION, "MEAN_STD")).split(".")[-1]
    stats = dataset.meta.stats["action"]
    if "MIN_MAX" in mode:
        return "MIN_MAX", np.asarray(stats["min"], np.float64), np.asarray(stats["max"], np.float64)
    return "MEAN_STD", np.asarray(stats["mean"], np.float64), np.asarray(stats["std"], np.float64)


def decode(norm: np.ndarray, mode: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if mode == "MIN_MAX":
        return (norm + 1.0) / 2.0 * (b - a) + a
    return norm * b + a


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    policy = ACTPolicy.from_pretrained(args.model)
    policy.to(device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=args.model)

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    image_keys = list(policy.config.image_features)
    action_names = dataset.meta.features["action"]["names"]
    if isinstance(action_names, dict):
        action_names = list(action_names.get("motors", action_names))
    gripper_idx = next((i for i, n in enumerate(action_names) if "gripper" in n.lower()),
                       len(action_names) - 1)
    latent_dim = policy.config.latent_dim
    mode, lo, hi = normalization_stats(policy, dataset)
    print(f"latent_dim={latent_dim}  K={args.samples}  action_norm={mode}  cameras={image_keys}")

    # Inject a chosen latent into the module that consumes it: in ACT.forward the
    # latent reaches the transformer only through encoder_latent_input_proj.
    holder: dict[str, torch.Tensor | None] = {"z": None}

    def pre_hook(_module, _inputs):
        return (holder["z"],) if holder["z"] is not None else None

    policy.model.encoder_latent_input_proj.register_forward_pre_hook(pre_hook)

    def predict(batch, z: torch.Tensor) -> np.ndarray:
        holder["z"] = z
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
        holder["z"] = None
        return decode(chunk[0, 0].to("cpu", torch.float64).numpy(), mode, lo, hi)

    hf = dataset.hf_dataset
    ep_actions: dict[int, list[np.ndarray]] = {}
    ep_global_idx: dict[int, list[int]] = {}
    for i in range(len(hf)):
        row = hf[i]
        ep = int(row["episode_index"])
        ep_actions.setdefault(ep, []).append(np.asarray(row["action"], np.float64))
        ep_global_idx.setdefault(ep, []).append(i)
    episodes = sorted(ep_actions)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    err_zero: list[float] = []
    err_best: list[float] = []
    err_meank: list[float] = []
    spreads: list[np.ndarray] = []
    per_joint_zero: list[np.ndarray] = []
    per_joint_best: list[np.ndarray] = []
    n_frames = 0

    for ep in episodes:
        actions = np.stack(ep_actions[ep])
        g = actions[:, gripper_idx]
        span = float(g.max() - g.min())
        g_norm = (g - g.min()) / span if span > 1e-6 else np.zeros_like(g)
        closed = g_norm < 0.5
        releases = [t for t in range(1, len(actions)) if (not closed[t]) and closed[t - 1]]
        frames = sorted({t + d for t in releases
                         for d in range(-args.release_window, args.release_window + 1)
                         if 0 <= t + d < len(actions)})

        for t in frames:
            item = dataset[ep_global_idx[ep][t]]
            obs = {k: item[k] for k in image_keys + ["observation.state"]}
            if "task" in item:
                obs["task"] = item["task"]
            batch = preprocessor(obs)
            expert = actions[t]

            a_zero = predict(batch, torch.zeros(1, latent_dim, device=device))
            sampled = np.stack([predict(batch, torch.randn(1, latent_dim, device=device))
                                for _ in range(args.samples)])

            e_zero = np.abs(a_zero - expert)
            e_sampled = np.abs(sampled - expert)          # (K, A)
            best_k = int(e_sampled.mean(axis=1).argmin())

            err_zero.append(float(e_zero.mean()))
            err_best.append(float(e_sampled.mean(axis=1).min()))
            err_meank.append(float(np.abs(sampled.mean(axis=0) - expert).mean()))
            spreads.append(sampled.std(axis=0))
            per_joint_zero.append(e_zero)
            per_joint_best.append(e_sampled[best_k])
            n_frames += 1
            if n_frames % 25 == 0:
                print(f"  ...{n_frames} release frames")

    if not n_frames:
        print("no release events found")
        return 1

    mae_zero = float(np.mean(err_zero))
    mae_best = float(np.mean(err_best))
    mae_meank = float(np.mean(err_meank))
    spread = np.stack(spreads).mean(axis=0)
    pj_zero = np.stack(per_joint_zero).mean(axis=0)
    pj_best = np.stack(per_joint_best).mean(axis=0)
    improvement = (mae_zero - mae_best) / mae_zero * 100.0 if mae_zero else 0.0

    print(f"\n==== latent probe: {Path(args.model).parts[-3]} ====")
    print(f"release frames    : {n_frames}   K = {args.samples}")
    print(f"MAE with z = 0    : {mae_zero:.3f}   <- what deployment does")
    print(f"MAE best-of-K     : {mae_best:.3f}   ({improvement:+.1f}% vs z=0)")
    print(f"MAE mean-of-K     : {mae_meank:.3f}   (sanity: should track z=0)")
    print("\nper-joint, averaged over frames:")
    print(f"  {'joint':<18}{'spread':>9}{'z=0 err':>10}{'best-K err':>12}")
    for name, s, a, b in zip(action_names, spread, pj_zero, pj_best):
        print(f"  {name:<18}{s:9.3f}{a:10.3f}{b:12.3f}")

    if improvement > 25 and float(spread.mean()) > 0.1:
        verdict = ("z ENCODES THE MODE: some latent recovers the expert action, so the decoder "
                   "can emit it and zeroing z at inference is discarding the choice")
    elif float(spread.mean()) < 0.05:
        verdict = ("z IS INERT: the decoder ignores the latent entirely, so the target was never "
                   "learned from the observation and no inference-time change can fix it")
    else:
        verdict = ("INCONCLUSIVE: the latent moves the output but does not recover the expert "
                   "action; report the numbers rather than a mechanism")
    print(f"\nverdict: {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "model": args.model,
            "repo_id": args.repo_id,
            "release_frames": n_frames,
            "samples": args.samples,
            "mae_z0": round(mae_zero, 4),
            "mae_best_of_k": round(mae_best, 4),
            "mae_mean_of_k": round(mae_meank, 4),
            "best_of_k_improvement_pct": round(improvement, 2),
            "mean_spread": round(float(spread.mean()), 4),
            "per_joint": {name: {"spread": round(float(s), 4),
                                 "z0": round(float(a), 4),
                                 "best_k": round(float(b), 4)}
                          for name, s, a, b in zip(action_names, spread, pj_zero, pj_best)},
        }, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
