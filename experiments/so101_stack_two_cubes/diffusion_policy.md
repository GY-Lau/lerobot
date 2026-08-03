# Diffusion Policy Comparison

The Diffusion Policy comparison uses exactly the same 30-episode dataset and
the same physical evaluation protocol as the ACT baseline. The primary matched
training run uses 30,000 optimizer updates and batch size 2, so both policies
consume 60,000 sampled training examples. Policy-specific architecture,
optimizer, scheduler, and action-horizon defaults remain unchanged and are
captured in each checkpoint's `train_config.json`.

## Jetson compatibility result

Validated on 2026-08-03 with Jetson Orin NX 16 GB, JetPack 6.2.2, LeRobot's
Python 3.10 environment, and the local dataset:

- Default Diffusion Policy: 262,952,742 trainable parameters.
- Native 640 x 480 front-camera frames; no image transforms.
- A 20-step, batch-size-1 run completed and its checkpoint reloaded as a
  `DiffusionPolicy` with all 262,952,742 parameters.
- Checkpoint size was approximately 3.0 GB: 1.05 GB model plus 2.10 GB optimizer
  state and small metadata files.
- A separate batch-size-2 AMP run completed without out-of-memory errors.
- After CUDA warm-up, batch-size-2 AMP updates took approximately 1.18 seconds,
  implying roughly 9.8 hours for 30,000 updates before checkpoint overhead.

These are compatibility and throughput measurements, not learning results. The
short-run losses must not be compared with the trained ACT loss.

## Reproduce the matched training run

Preview the exact command without starting training:

```bash
bash experiments/so101_stack_two_cubes/train_diffusion.sh --dry-run \
  diffusion_stack_two_cubes_30k 30000 2 true 5000
```

Start it directly in the foreground:

```bash
bash experiments/so101_stack_two_cubes/train_diffusion.sh \
  diffusion_stack_two_cubes_30k 30000 2 true 5000
```

The expected final model is:

```text
outputs/train/diffusion_stack_two_cubes_30k/checkpoints/030000/pretrained_model
```

Do not report the run as an ACT-versus-Diffusion result until the 30,000-step
checkpoint has completed and both policies have run the unchanged 30-trial
physical schedule from `evaluation_protocol.md`.
