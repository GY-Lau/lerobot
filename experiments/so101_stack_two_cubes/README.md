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

## Directory layout

```text
so101_stack_two_cubes/
├── README.md              # entry point and artifact index
├── PROJECT_REPORT.md      # portfolio-facing evidence report
├── docs/                  # protocols, audits, and experiment interpretation
├── manifests/             # subset contracts and placement metadata
├── assets/                # static workspace assets
├── results/               # measured CSV/TXT outputs and trial records
├── scripts/               # recording, training, evaluation, and audit CLIs
└── tests/                 # experiment-level regression tests
```

Generated Python caches and model checkpoints do not belong here. Checkpoints
remain under the repository-level `outputs/` directory.

## Planned comparisons

1. ACT data and training-step scaling.
2. ACT versus Diffusion Policy on the same data and evaluation protocol.
3. SmolVLA LoRA fine-tuning for language-conditioned task selection.
4. Jetson camera-to-action latency under AMP and power-mode variants.

## Reproducibility artifacts

- [`PROJECT_REPORT.md`](PROJECT_REPORT.md): portfolio-facing evidence report
  with verified results, unsupported-claim boundaries, and remaining gates.
- [`evaluation_protocol.md`](docs/evaluation_protocol.md): success definition,
  placement regimes, and failure taxonomy.
- [`data_audit.md`](docs/data_audit.md): structural, start-pose, and visual audit of
  the original 30 demonstrations.
- [`act_data_efficiency.md`](docs/act_data_efficiency.md): controlled 10/20/30
  episode ACT ablation and matched-budget interpretation rules.
- [`act_v2.md`](docs/act_v2.md): independent clean 50-episode collection, nested
  30/50 subset audit, and matched ACT improvement protocol.
- [`training_start_positions.csv`](manifests/training_start_positions.csv): per-episode
  marker-free cube detections used to construct the ACT subsets.
- [`act_data_subsets.json`](manifests/act_data_subsets.json): deterministic nested episode
  membership with source-data hash and quality balance.
- [`generate_act_subsets.py`](scripts/generate_act_subsets.py): reproducible spatial
  coverage and quality-stratified subset generator.
- [`train_act_data_efficiency.sh`](scripts/train_act_data_efficiency.sh): guarded ACT
  10/20/30 episode training launcher matched to the 30-episode baseline.
- [`verify_act_checkpoint.py`](scripts/verify_act_checkpoint.py): validates final files,
  training step, subset membership, and the matched ACT training contract.
- [`act_training_runs.csv`](results/act_training_runs.csv): completed 10/20/30-episode
  matched-budget ACT runs, retained loss sources, and checkpoint verification.
- [`inventory_model_artifacts.py`](scripts/inventory_model_artifacts.py): checks
  completed training steps and hashes model weights plus training configs.
- [`model_artifacts.csv`](results/model_artifacts.csv): Jetson-generated sizes and
  SHA-256 identities for all completed ACT and Diffusion checkpoints.
- [`results.csv`](results/results.csv): append-only experiment summary.
- [`trials.csv`](results/trials.csv): one auditable row per physical evaluation trial.
- [`check_start_pose.py`](scripts/check_start_pose.py): read-only validation of the
  follower's initial joint pose.
- [`log_trial.py`](scripts/log_trial.py): validated, append-only entry of one physical
  trial at a time.
- [`summarize_trials.py`](scripts/summarize_trials.py): success rates, Wilson 95%
  confidence intervals, and failure counts.
- [`summarize_act_screening.py`](scripts/summarize_act_screening.py): joins the complete
  five-trial-per-checkpoint ACT screen with its measured latency logs.
- [`act_physical_screening.csv`](results/act_physical_screening.csv): machine-readable
  exploratory outcomes, confidence intervals, failures, and control latency.
- [`summarize_latency.py`](scripts/summarize_latency.py): mean/P50/P95 control latency,
  deadline misses, and action-chunk refresh versus cached-action timing.
- [`run_act_trial.sh`](scripts/run_act_trial.sh): pose-gated, one-episode ACT evaluation
  with automatic dataset resume.
- [`run_act_data_efficiency_trial.sh`](scripts/run_act_data_efficiency_trial.sh): maps
  10/20/30 episodes to the verified checkpoint and a distinct evaluation run.
- [`analyze_cube_placements.py`](scripts/analyze_cube_placements.py): marker-free red
  and yellow cube localization in normalized camera coordinates.
- [`diffusion_policy.md`](docs/diffusion_policy.md): matched-comparison design and
  verified Jetson training and closed-loop inference measurements.
- [`jetson_diffusion_latency.csv`](results/jetson_diffusion_latency.csv): auditable
  n2/n5/n10/n100 Diffusion latency comparison with raw-log hashes.
- [`train_diffusion.sh`](scripts/train_diffusion.sh): guarded, reproducible Diffusion
  Policy training command.
- [`smolvla_peft.md`](docs/smolvla_peft.md): evidence-gated two-task language-control
  design, PEFT training protocol, and text-before-voice evaluation matrix.
- [`record_inverse_language_data.sh`](scripts/record_inverse_language_data.sh):
  progress-aware one-episode recorder for the red-on-yellow inverse behavior;
  it stops at the declared 30-episode target.
- [`record_act_v2_data.sh`](scripts/record_act_v2_data.sh): progress-aware recorder for
  the independent 50-episode, higher-quality yellow-on-red ACT v2 dataset.
- [`record_act_v2_session.py`](scripts/record_act_v2_session.py): persistent interactive
  ACT v2 recorder that pays LeRobot's import cost once per recording session.
- [`discard_last_act_v2_episode.sh`](scripts/discard_last_act_v2_episode.sh): guarded
  removal of a failed final v2 demonstration with the original data retained.
- [`prepare_act_v2_subsets.sh`](scripts/prepare_act_v2_subsets.sh): audits all 50 v2
  start frames and generates spatially balanced nested 30/50 subsets.
- [`train_act_v2.sh`](scripts/train_act_v2.sh): matched-contract ACT v2 trainer for the
  audited 30- and 50-episode subsets.
- [`prepare_language_dataset.sh`](scripts/prepare_language_dataset.sh): guarded merge
  of the two color orders into one balanced multi-task dataset.
- [`validate_language_dataset.py`](scripts/validate_language_dataset.py): verifies
  task vocabulary, per-task episode balance, and label integrity.
- [`train_smolvla_peft.sh`](scripts/train_smolvla_peft.sh): guarded SmolVLA LoRA smoke
  test and full-training launcher.
- [`jetson_smolvla_environment.txt`](results/jetson_smolvla_environment.txt): validated
  JetPack, Python, CUDA/BF16, and PEFT package versions.
- [`generate_trial_plan.py`](scripts/generate_trial_plan.py): deterministic placement
  schedule generator retained for setups that use physical coordinates.
