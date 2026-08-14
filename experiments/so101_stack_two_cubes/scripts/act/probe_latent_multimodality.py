#!/usr/bin/env python3
"""Does ACT's CVAE latent carry the task-relevant mode, or is it vacuous?

At inference ACT sets z to zero, so the decoder returns the conditional mean of
the action distribution. If a dataset contains two valid targets for the same
observation, that mean can be an invalid action. This probe asks whether the
information needed to pick the right target is *in z* (recoverable by sampling)
or was never learned at all.

For each evaluated frame it compares, against the expert action:

  * z = 0            -- exactly what deployment does;
  * K samples z~N(0,I) -- best-of-K, mean-of-K, and the spread across samples.

Readings:
  best-of-K much better than z=0  -> z encodes the mode; averaging is the
                                     inference-time zeroing, and the decoder
                                     *can* produce the right action.
  best-of-K ~= z=0, spread ~= 0   -> z is vacuous; the decoder never learned to
                                     read the target from pixels.

The latent is injected with a forward pre-hook on the module that consumes it
(`encoder_latent_input_proj`), so nothing in the repo is modified.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=32, help="K latent draws per frame")
    p.add_argument("--release-window", type=int, default=2,
                   help="frames each side of the expert release to evaluate")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default=None, help="optional JSON summary path")
    return p.parse_args()


def norm_stats(policy, dataset):
    from lerobot.configs.types import FeatureType
    mode = str(policy.config.normalization_mapping.get(FeatureType.ACTION, "MEAN_STD")).split(".")[-1]
    s = dataset.meta.stats["action"]
    if "MIN_MAX" in mode:
        return "MIN_MAX", np.asarray(s["min"], float), np.asarray(s["max"], float)
    return "MEAN_STD", np.asarray(s["mean"], float), np.asarray(s["std"], float)


def decode(a, mode, lo, hi):
    return (a + 1.0) / 2.0 * (hi - lo) + lo if mode == "MIN_MAX" else a * hi + lo


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    policy = ACTPolicy.from_pretrained(args.model)
    policy.to(device).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=args.model)

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    image_keys = list(policy.config.image_features)
    names = dataset.meta.features["action"]["names"]
    if isinstance(names, dict):
        names = list(names.get("motors", names))
    gi = next((i for i, n in enumerate(names) if "gripper" in n.lower()), len(names) - 1)
    latent_dim = policy.config.latent_dim
    mode, lo, hi = norm_stats(policy, dataset)
    print(f"latent_dim={latent_dim}  K={args.samples}  action_norm={mode}  cameras={image_keys}")

    # Inject a chosen latent into the module that consumes it.
    holder: dict[str, torch.Tensor | None] = {"z": None}

    def pre_hook(_module, inputs):
        return (holder["z"],) if holder["z"] is not None else None

    policy.model.encoder_latent_input_proj.register_forward_pre_hook(pre_hook)

    def predict(batch, z):
        holder["z"] = z
        with torch.inference_mode():
            out = policy.predict_action_chunk(batch)
        holder["z"] = None
        return decode(out[0, 0].to("cpu", torch.float64).numpy(), mode, lo, hi)

    # Locate expert release events (gripper opens) per episode.
    hf = dataset.hf_dataset
    ep_actions: dict[int, list[np.ndarray]] = {}
    ep_idx: dict[int, list[int]] = {}
    for i in range(len(hf)):
        r = hf[i]
        e = int(r["episode_index"])
        ep_actions.setdefault(e, []).append(np.asarray(r["action"], float))
        ep_idx.setdefault(e, []).append(i)
    episodes = sorted(ep_actions)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    err_zero, err_best, err_meanK, spreads = [], [], [], []
    per_joint_zero, per_joint_best = [], []
    n = 0
    for ep in episodes:
        acts = np.stack(ep_actions[ep])
        g = acts[:, gi]
        span = float(g.max() - g.min())
        gn = (g - g.min()) / span if span > 1e-6 else np.zeros_like(g)
        closed = gn < 0.5
        releases = [t for t in range(1, len(acts)) if (not closed[t]) and closed[t - 1]]
        frames = sorted({t + d for t in releases
                         for d in range(-args.release_window, args.release_window + 1)
                         if 0 <= t + d < len(acts)})
        for t in frames:
            item = dataset[ep_idx[ep][t]]
            obs = {k: item[k] for k in image_keys + ["observation.state"]}
            if "task" in item:
                obs["task"] = item["task"]
            batch = pre(obs)
            expert = acts[t]

            a0 = predict(batch, torch.zeros(1, latent_dim, device=device))
            samples = np.stack([
                predict(batch, torch.randn(1, latent_dim, device=device))
                for _ in range(args.samples)
            ])

            e0 = np.abs(a0 - expert)
            eK = np.abs(samples - expert)             # (K, A)
            best_k = int(eK.mean(axis=1).argmin())

            err_zero.append(e0.mean())
            err_best.append(eK.mean(axis=1).min())
            err_meanK.append(np.abs(samples.mean(0) - expert).mean())
            spreads.append(samples.std(axis=0))
            per_joint_zero.append(e0)
            per_joint_best.append(eK[best_k])
            n += 1
            if n % 25 == 0:
                print(f"  ...{n} release frames")

    z0 = float(np.mean(err_zero))
    bk = float(np.mean(err_best))
    mk = float(np.mean(err_meanK))
    spread = np.stack(spreads).mean(axis=0)
    pj_zero = np.stack(per_joint_zero).mean(axis=0)
    pj_best = np.stack(per_joint_best).mean(axis=0)
    drop = (z0 - bk) / z0 * 100 if z0 else 0.0

    print(f"\n==== latent probe: {Path(args.model).parts[-3]} ====")
    print(f"release frames evaluated : {n}   K = {args.samples}")
    print(f"MAE with z = 0           : {z0:.3f}   <- what deployment does")
    print(f"MAE best-of-K sampled z  : {bk:.3f}   ({drop:+.1f}% vs z=0)")
    print(f"MAE mean-of-K sampled z  : {mk:.3f}   (sanity: should track z=0)")
    print("\nper-joint spread across z samples (std, raw units):")
    for nm, s, a, b in zip(names, spread, pj_zero, pj_best):
        print(f"  {nm:<18} spread={s:7.3f}   z=0 err={a:7.3f}   best-K err={b:7.3f}")

    verdict = ("z ENCODES THE MODE - sampling recovers the expert action, so the "
               "decoder can produce it and inference-time zeroing is discarding it"
               if drop > 25 and spread.mean() > 0.1 else
               "z LOOKS VACUOUS - sampling does not help, so the target was never "
               "learned from the observation")
    print(f"\nverdict: {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "model": args.model, "repo_id": args.repo_id, "frames": n,
            "samples": args.samples,
            "mae_z0": round(z0, 4), "mae_best_of_k": round(bk, 4),
            "mae_mean_of_k": round(mk, 4), "best_of_k_improvement_pct": round(drop, 2),
            "per_joint": {nm: {"spread": round(float(s), 4),
                               "z0": round(float(a), 4), "best_k": round(float(b), 4)}
                          for nm, s, a, b in zip(names, spread, pj_zero, pj_best)},
        }, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
