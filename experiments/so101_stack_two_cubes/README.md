# SO-101 Two-Cube Stacking

This experiment studies data efficiency, policy choice, and edge deployment for
vision-based manipulation on a low-cost SO-101 arm.

## Task

Use a fixed front RGB camera and joint observations to stack the yellow cube on
top of the red cube.

## Current baseline

| Item | Value |
| --- | --- |
| Robot | SO-101 follower |
| Compute | NVIDIA Jetson Orin NX 16 GB |
| Power mode | 25 W |
| Camera | Front RGB, 640 x 480, 30 FPS |
| Dataset | `GY-William/lerobot_stack_two_cubes` |
| Demonstrations | 30 episodes / 17,970 frames |
| Policy | ACT, ResNet-18, action chunk 100 |
| Training | 30,000 optimizer steps, batch size 2 |
| Final training loss | 0.173 |
| Exploratory evaluation | 2/10 successes at a 30 s horizon |

The exploratory 2/10 result is not the official baseline because the evaluation
horizon was longer than the approximately 20 s demonstrations and the cube
placement protocol was not yet standardized.

## Planned comparisons

1. ACT data and training-step scaling.
2. ACT versus Diffusion Policy on the same data and evaluation protocol.
3. SmolVLA LoRA fine-tuning for language-conditioned task selection.
4. Jetson camera-to-action latency under AMP and power-mode variants.

## Reproducibility artifacts

- [`evaluation_protocol.md`](evaluation_protocol.md): success definition,
  placement regimes, and failure taxonomy.
- [`results.csv`](results.csv): append-only experiment summary.
- [`trials.csv`](trials.csv): one auditable row per physical evaluation trial.
