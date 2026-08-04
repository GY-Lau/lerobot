# Diffusion Policy Comparison

The Diffusion Policy comparison uses exactly the same 30-episode dataset and
the same physical evaluation protocol as the ACT baseline. The primary matched
training run uses 30,000 optimizer updates and batch size 2, so both policies
consume 60,000 sampled training examples. Policy-specific architecture,
optimizer, scheduler, and action-horizon defaults remain unchanged and are
captured in each checkpoint's `train_config.json`.

## Jetson training compatibility

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

These are training compatibility and throughput measurements, not learning
results. The short-run losses must not be compared with the trained ACT loss.

## Jetson closed-loop inference result

The matched 30,000-step model completed training, but the stock synchronous
`lerobot-record` control loop cannot run it at 30 FPS on the Jetson. When the
Diffusion action queue is empty, the control thread blocks while the complete
action chunk is denoised. Cached actions are fast, but they are delivered in a
short burst after each blocking refresh.

One physical screening trial was run for each inference setting on 2026-08-04.
These trials diagnose deployment latency; they are not success-rate estimates
and must not be reported as the matched policy comparison.

| Reverse steps | Action steps | AMP | Refresh policy P50 | Effective FPS | Physical observation |
| ---: | ---: | :---: | ---: | ---: | --- |
| 100 | 8 | no | 21,085.95 ms | 0.047 | First chunk exceeded the 20 s episode; no observable task motion |
| 10 | 8 | yes | 1,835.15 ms | 3.408 | Long pause followed by a short action burst |
| 5 | 8 | yes | 973.89 ms | 6.268 | Pause-burst motion; more motion than the two-step run |
| 2 | 15 | yes | 466.48 ms | 15.081 | Jerky motion and no grasp |

`effective FPS` is `recorded frames / sum(loop_period_ms)`, not the warning's
instantaneous rate. Every action-chunk refresh missed the 33.33 ms deadline;
cached frames had approximately 10.2--10.7 ms policy latency and ran at the
30 FPS loop period. Reducing reverse steps therefore trades denoising quality
for speed but does not remove the synchronous pause.

The machine-readable summary, including raw-log SHA-256 hashes, is stored in
[`jetson_diffusion_latency.csv`](jetson_diffusion_latency.csv). Recompute a
single raw log with:

```bash
python experiments/so101_stack_two_cubes/summarize_latency.py \
  outputs/eval_latency/eval_diffusion_30k_n5_amp_fixed_30s.csv \
  --include-warmup
```

### Deployment decision

- Keep ACT as the real-time Jetson baseline.
- Preserve the trained Diffusion model as the same-data policy comparison, but
  do not claim real-time Jetson deployment with the current 263M-parameter
  architecture and synchronous controller.
- Do not extend the episode horizon to hide inference latency. The task and
  success definition remain fixed at 20 seconds.
- A future deployable Diffusion variant requires a smaller network and/or
  asynchronous action generation. It is a separate optimization experiment,
  not a post-hoc change to the matched model.

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

The 30,000-step checkpoint is complete. Do not report the run as an
ACT-versus-Diffusion success-rate result until both policies have run the
unchanged physical schedule from `evaluation_protocol.md`. The latency screen
above shows that this requires either non-Jetson Diffusion inference or a
separately disclosed deployment implementation; reduced-step screening runs
are not interchangeable with the trained default policy.
