# SO-101 Two-Cube Stacking

Imitation learning on a $500 SO-101 arm, trained on a workstation and deployed on
a Jetson Orin NX. Two policy families (ACT and SmolVLA), 55 logged physical
trials, and a report that separates what the evidence supports from what it does
not.

**Task**: stack the yellow cube on the red cube, from a wrist camera, a fixed
front camera, and joint positions.

The success rates here are modest. The point of the project is the diagnostics —
each failure was traced to a cause by measurement, and three of the causes turned
out not to be the model.

## What this found

Full evidence for each in [`PROJECT_REPORT.md`](PROJECT_REPORT.md)
([中文版](PROJECT_REPORT.zh-CN.md)).

1. **Observability is a property of the whole rig, not the camera.** Adding a
   second camera made the policy worse, because mounting it forced the gripper
   horizontal and cost the side view that carries grasp height. Fixing it changed
   no model code.
2. **Open-loop fitting error ranks policies backwards for deployment.** The
   checkpoint that fit demonstrations best (1.293 deg) scored **0/6**. Measured
   three times, most sharply on a single checkpoint against both of its training
   layouts: **1.585 vs 1.304 deg, and 0/5 vs 2/5**.
3. **A deployment parameter beat every model change attempted.** At the inherited
   `n_action_steps=20` the arm stalled — re-executing the head of each fresh
   chunk, never reaching its descend-and-close tail. Setting it to 100, same
   weights, moved grasping **0/5 to 4/5**.
4. **More data is not automatically better.** Merging 20 episodes whose cube
   positions did not overlap the evaluation layout took the policy **2/5 to 0/6**,
   and changed the failure mode to releasing short of the target. Confirmed by
   two independent measurements, not by trial counts.
5. **ACT's CVAE is posterior-collapsed here.** Sampling the latent 32 times beat
   `z = 0` by **0.6%**, with under 0.04 deg of spread. At `kl_weight = 10.0` on
   20–40 episodes it is a plain L1 regressor, so the averaging in #4 happens in
   the decoder and no latent-side hyperparameter can fix it.
6. **Evaluation conditions are part of the experiment.** Same checkpoint: **5/5**
   grasps with the room light on, **0/3** with it off. The largest single-factor
   effect measured here came from an uncontrolled variable.
7. **No controller is neutral between these two policies.** Their
   inference-to-chunk ratios differ by about **80x**. Run synchronously SmolVLA
   is frozen 43% of the time and scores 0/5; asynchronously, 3/5. Running both
   families under one controller relocates the confound instead of removing it.
8. **Sampling removed the averaging but not the failure.** SmolVLA on the merge
   that cost ACT everything scored 2/5 with none of ACT's release-short
   signature — yet 0/5 in the other layout it was equally trained on. Mode
   collapse is excluded: fed expert observations it fits that layout to within
   22% of the one it succeeds in.

## Setup

| | |
| --- | --- |
| Robot | SO-101 follower, vertical gripper |
| Cameras | Wrist (mounted 90 deg from upright) + fixed front, 640x480, 30 FPS |
| Inference | Jetson Orin NX 16 GB, MAXN_SUPER + `jetson_clocks` |
| Training | 2x RTX A4500 20 GB |
| Datasets | `vertical_redleft_20ep`, `vertical_20ep`, and their 40-episode merge |
| ACT | ResNet-18, chunk 100, 30k steps, batch 8, seed 1000, AMP off |
| SmolVLA | 450M base, released config unmodified, chunk 50, async inference |

ACT's 30k steps is a departure from LeRobot's 100k default, held constant across
every ACT comparison. Any absolute ACT number should be read with that in mind.
SmolVLA's own config happens to specify 30k steps and batch 8, so the two
families match on steps, batch, seed, and data exposure without either being
bent to fit the other.

## Physical results

One auditable row per trial in [`trials.csv`](results/trials.csv);
`summarize_trials.py` reproduces every rate and Wilson interval from the file.

| Run | Result | Notes |
| --- | ---: | --- |
| ACT 10 / 20 / 30 episodes | 2/5, 3/5, 0/5 | data-efficiency screen, matched budget |
| ACT red-left, `n_action_steps=100` | 2/8 | 3 of the 8 ran under the unlit condition of #6 |
| ACT merged 40 episodes | 0/6 | finding #4 |
| ACT red-left, async | 1/5 | 3 timeouts; async hurts the fast policy |
| SmolVLA red-left, sync | 0/5 | finding #7 |
| SmolVLA red-left, async | **3/5** | best result measured |
| SmolVLA merged, async | 2/5 | no release-short failures |
| SmolVLA merged, swapped layout | 0/5 | finding #8 |

No pair here separates statistically at these sample sizes, and the report says
so where it matters.

## Where to look

- [`PROJECT_REPORT.md`](PROJECT_REPORT.md) — the evidence, including an
  **evidence matrix**, a list of **claims the data does not support**, and the
  next experiments ranked by what would change a conclusion.
- [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) — success
  definition, placement regimes, failure taxonomy.
- [`results/trials.csv`](results/trials.csv) — every physical trial.
- [`docs/artifact_index.md`](docs/artifact_index.md) — the full script and
  output index, one entry per reproducibility artifact.

## Directory layout

```text
so101_stack_two_cubes/
├── README.md              # this file
├── PROJECT_REPORT.md      # evidence report (+ .zh-CN.md)
├── docs/                  # protocols, audits, artifact index
├── manifests/             # subset contracts, pinned model hashes
├── assets/                # static workspace assets
├── results/               # measured outputs and trial records
├── scripts/               # act/ diffusion/ smolvla/ common/
└── tests/                 # experiment-level regression tests
```

Checkpoints stay under the repository-level `outputs/`, not here.

## Not yet done

A language-conditioned experiment. SmolVLA is a vision-**language**-action model
and every trial here used a single instruction, so nothing in this report tests
the language channel. The design is in
[`docs/smolvla_peft.md`](docs/smolvla_peft.md) and needs 30 new demonstrations of
a second, physically distinct task.
