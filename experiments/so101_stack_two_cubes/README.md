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
- [`data_audit.md`](data_audit.md): structural, start-pose, and visual audit of
  the original 30 demonstrations.
- [`results.csv`](results.csv): append-only experiment summary.
- [`trials.csv`](trials.csv): one auditable row per physical evaluation trial.
- [`check_start_pose.py`](check_start_pose.py): read-only validation of the
  follower's initial joint pose.
- [`log_trial.py`](log_trial.py): validated, append-only entry of one physical
  trial at a time.
- [`summarize_trials.py`](summarize_trials.py): success rates, Wilson 95%
  confidence intervals, and failure counts.
- [`summarize_latency.py`](summarize_latency.py): mean/P50/P95 control latency,
  deadline misses, and action-chunk refresh versus cached-action timing.
- [`run_act_trial.sh`](run_act_trial.sh): pose-gated, one-episode ACT evaluation
  with automatic dataset resume.
- [`analyze_cube_placements.py`](analyze_cube_placements.py): marker-free red
  and yellow cube localization in normalized camera coordinates.
- [`diffusion_policy.md`](diffusion_policy.md): matched-comparison design and
  verified Jetson training and closed-loop inference measurements.
- [`jetson_diffusion_latency.csv`](jetson_diffusion_latency.csv): auditable
  n2/n5/n10/n100 Diffusion latency comparison with raw-log hashes.
- [`train_diffusion.sh`](train_diffusion.sh): guarded, reproducible Diffusion
  Policy training command.
- [`smolvla_peft.md`](smolvla_peft.md): evidence-gated two-task language-control
  design, PEFT training protocol, and text-before-voice evaluation matrix.
- [`record_inverse_language_data.sh`](record_inverse_language_data.sh):
  one-episode recorder for the red-on-yellow inverse behavior.
- [`prepare_language_dataset.sh`](prepare_language_dataset.sh): guarded merge
  of the two color orders into one balanced multi-task dataset.
- [`validate_language_dataset.py`](validate_language_dataset.py): verifies
  task vocabulary, per-task episode balance, and label integrity.
- [`train_smolvla_peft.sh`](train_smolvla_peft.sh): guarded SmolVLA LoRA smoke
  test and full-training launcher.
- [`generate_trial_plan.py`](generate_trial_plan.py): deterministic placement
  schedule generator retained for setups that use physical coordinates.
