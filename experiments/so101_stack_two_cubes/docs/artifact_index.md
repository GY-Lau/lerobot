# Reproducibility artifact index

Every measurement in [`PROJECT_REPORT.md`](../PROJECT_REPORT.md) traces to one of
these. Scripts are runnable CLIs; CSV and TXT files are their committed outputs.

- [`PROJECT_REPORT.md`](../PROJECT_REPORT.md): portfolio-facing evidence report
  with verified results, unsupported-claim boundaries, and remaining gates.
- [`evaluation_protocol.md`](evaluation_protocol.md): success definition,
  placement regimes, and failure taxonomy.
- [`data_audit.md`](data_audit.md): structural, start-pose, and visual audit of
  the original 30 demonstrations.
- [`act_data_efficiency.md`](act_data_efficiency.md): controlled 10/20/30
  episode ACT ablation and matched-budget interpretation rules.
- [`act_v2.md`](act_v2.md): independent clean 50-episode collection, nested
  30/50 subset audit, and matched ACT improvement protocol.
- [`a4500_act_v2_environment.txt`](../results/a4500_act_v2_environment.txt): exact
  source, package, GPU, dataset, and training provenance for the isolated
  A4500 v2-50 run.
- [`training_start_positions.csv`](../manifests/training_start_positions.csv): per-episode
  marker-free cube detections used to construct the ACT subsets.
- [`act_data_subsets.json`](../manifests/act_data_subsets.json): deterministic nested episode
  membership with source-data hash and quality balance.
- [`generate_act_subsets.py`](../scripts/act/generate_act_subsets.py): reproducible spatial
  coverage and quality-stratified subset generator.
- [`train_act_data_efficiency.sh`](../scripts/act/train_act_data_efficiency.sh): guarded ACT
  10/20/30 episode training launcher matched to the 30-episode baseline.
- [`verify_act_checkpoint.py`](../scripts/act/verify_act_checkpoint.py): validates final files,
  training step, subset membership, and the matched ACT training contract.
- [`act_training_runs.csv`](../results/act_training_runs.csv): completed 10/20/30-episode
  matched-budget ACT runs, retained loss sources, and checkpoint verification.
- [`inventory_model_artifacts.py`](../scripts/common/inventory_model_artifacts.py): checks
  completed training steps and hashes model weights plus training configs.
- [`model_artifacts.csv`](../results/model_artifacts.csv): Jetson-generated sizes and
  SHA-256 identities for all completed ACT and Diffusion checkpoints.
- [`results.csv`](../results/results.csv): append-only experiment summary.
- [`trials.csv`](../results/trials.csv): one auditable row per physical evaluation trial.
- [`check_start_pose.py`](../scripts/common/check_start_pose.py): read-only validation of the
  follower's initial joint pose.
- [`log_trial.py`](../scripts/common/log_trial.py): validated, append-only entry of one physical
  trial at a time.
- [`summarize_trials.py`](../scripts/common/summarize_trials.py): success rates, Wilson 95%
  confidence intervals, and failure counts.
- [`summarize_act_screening.py`](../scripts/act/summarize_act_screening.py): joins the complete
  five-trial-per-checkpoint ACT screen with its measured latency logs.
- [`act_physical_screening.csv`](../results/act_physical_screening.csv): machine-readable
  exploratory outcomes, confidence intervals, failures, and control latency.
- [`summarize_latency.py`](../scripts/common/summarize_latency.py): mean/P50/P95 control latency,
  deadline misses, and action-chunk refresh versus cached-action timing.
- [`run_act_trial.sh`](../scripts/act/run_act_trial.sh): pose-gated, one-episode ACT evaluation
  with automatic dataset resume.
- [`run_act_data_efficiency_trial.sh`](../scripts/act/run_act_data_efficiency_trial.sh): maps
  10/20/30 episodes to the verified checkpoint and a distinct evaluation run.
- [`run_act_v2_trial.sh`](../scripts/act/run_act_v2_trial.sh): maps the clean 30/50 ACT v2
  subsets to distinct verified checkpoints and evaluation datasets.
- [`analyze_cube_placements.py`](../scripts/common/analyze_cube_placements.py): marker-free red
  and yellow cube localization in normalized camera coordinates.
- [`diffusion_policy.md`](diffusion_policy.md): matched-comparison design and
  verified Jetson training and closed-loop inference measurements.
- [`jetson_diffusion_latency.csv`](../results/jetson_diffusion_latency.csv): auditable
  n2/n5/n10/n100 Diffusion latency comparison with raw-log hashes.
- [`train_diffusion.sh`](../scripts/diffusion/train_diffusion.sh): guarded, reproducible Diffusion
  Policy training command.
- [`smolvla_peft.md`](smolvla_peft.md): evidence-gated two-task language-control
  design, PEFT training protocol, and text-before-voice evaluation matrix.
- [`record_inverse_language_data.sh`](../scripts/smolvla/record_inverse_language_data.sh):
  progress-aware one-episode recorder for the red-on-yellow inverse behavior;
  it stops at the declared 30-episode target.
- [`record_inverse_language_session.py`](../scripts/smolvla/record_inverse_language_session.py):
  persistent keep/discard recording loop for the inverse language behavior.
- [`prepare_inverse_language_audit.sh`](../scripts/smolvla/prepare_inverse_language_audit.sh):
  extracts all inverse-task start frames and applies visual-quality and
  cross-task role-aligned placement gates.
- [`validate_language_position_balance.py`](../scripts/smolvla/validate_language_position_balance.py):
  detects gross task-label/layout confounding using median shift and p10--p90 overlap.
- [`discard_last_dataset_episode.sh`](../scripts/common/discard_last_dataset_episode.sh): guarded,
  repository-parameterized last-episode removal with retained source backups.
- [`record_act_v2_data.sh`](../scripts/act/record_act_v2_data.sh): progress-aware recorder for
  the independent 50-episode, higher-quality yellow-on-red ACT v2 dataset.
- [`record_act_v2_session.py`](../scripts/common/record_act_v2_session.py): persistent interactive
  ACT v2 recorder that pays LeRobot's import cost once per recording session.
- [`discard_last_act_v2_episode.sh`](../scripts/act/discard_last_act_v2_episode.sh): guarded
  removal of a failed final v2 demonstration with the original data retained.
- [`prepare_act_v2_subsets.sh`](../scripts/act/prepare_act_v2_subsets.sh): audits all 50 v2
  start frames and generates spatially balanced nested 30/50 subsets.
- [`train_act_v2.sh`](../scripts/act/train_act_v2.sh): matched-contract ACT v2 trainer for the
  audited 30- and 50-episode subsets.
- [`train_act_v2_sequence.sh`](../scripts/act/train_act_v2_sequence.sh): guarded sequential
  30/50 trainer that verifies each final checkpoint before continuing.
- [`prepare_language_dataset.sh`](../scripts/smolvla/prepare_language_dataset.sh): guarded copy
  of the committed ACT v2 30-episode subset and merge with the inverse color order
  into one balanced multi-task dataset.
- [`smolvla_base.json`](../manifests/smolvla_base.json): immutable SmolVLA and
  SmolVLM2 processor/config revisions, required-file lists, source URLs, and
  expected SHA-256 identities.
- [`prepare_smolvla_base.sh`](../scripts/smolvla/prepare_smolvla_base.sh): resumable pinned
  snapshot download through the configured Hugging Face endpoint.
- [`verify_smolvla_base.py`](../scripts/smolvla/verify_smolvla_base.py): rejects missing,
  empty, or hash-mismatched base-model snapshots before PEFT training.
- [`validate_language_dataset.py`](../scripts/smolvla/validate_language_dataset.py): verifies
  task vocabulary, per-task episode balance, and label integrity.
- [`train_smolvla_peft.sh`](../scripts/smolvla/train_smolvla_peft.sh): guarded SmolVLA LoRA smoke
  test and full-training launcher.
- [`run_smolvla_language_trial.sh`](../scripts/smolvla/run_smolvla_language_trial.sh): fixed
  exact/paraphrase, two-order physical evaluation runner with latency logging.
- [`log_smolvla_language_trial.py`](../scripts/smolvla/log_smolvla_language_trial.py): records
  manipulation and instruction-following outcomes as separate audited metrics.
- [`summarize_smolvla_language_trials.py`](../scripts/smolvla/summarize_smolvla_language_trials.py):
  reports four-condition coverage, dual success rates, Wilson intervals, and failures.
- [`jetson_smolvla_environment.txt`](../results/jetson_smolvla_environment.txt): validated
  JetPack, Python, CUDA/BF16, and PEFT package versions.
- [`generate_trial_plan.py`](../scripts/common/generate_trial_plan.py): deterministic placement
  schedule generator retained for setups that use physical coordinates.
