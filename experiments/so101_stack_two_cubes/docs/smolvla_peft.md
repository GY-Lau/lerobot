# SmolVLA PEFT Language-Control Plan

## Evidence gate

The yellow-on-red source dataset was inspected and curated directly on
2026-08-06:

- `GY-William/lerobot_stack_two_cubes_v2`
- 50 retained episodes, 29,914 frames, 30 FPS
- exactly one task label: `Stack the yellow cube on top of the red cube`

The language experiment takes the committed, spatially balanced 30-episode
subset from `manifests/act_v2_subsets.json`. This keeps the two behaviors at
30 demonstrations each while using the manually reviewed v2 data rather than
the noisier original ACT dataset. It also prevents data quality from becoming
an accidental proxy for the requested color order.

A single behavior still cannot demonstrate language-conditioned task selection.
Relabeling the same trajectory with paraphrases would teach multiple strings for
the same action, not distinct language-controlled behaviors.

The reportable language experiment therefore uses two balanced behaviors:

1. `Stack the yellow cube on top of the red cube` (audited v2 subset, 30 episodes).
2. `Stack the red cube on top of the yellow cube` (30 new episodes).

Keep camera, background, arm calibration, 30 FPS, 20-second horizon, and cube
placement distribution consistent across both directions. Spread each task
over the same position variations; do not record all examples of one task in
one layout and the other task in another layout, because color order would be
confounded with position.

## 1. Record the inverse task

Use the persistent session recorder so LeRobot's approximately 25--30 second
import cost is paid once. It prompts before each episode and lets you keep or
discard the saved trajectory immediately, with the discarded source retained
as a backup:

```bash
PYTHONNOUSERSITE=1 /home/hai/miniconda3/envs/lerobot/bin/python \
  experiments/so101_stack_two_cubes/scripts/record_inverse_language_session.py
```

The original one-shot command remains available for status checks and as a
fallback:

```bash
bash experiments/so101_stack_two_cubes/scripts/record_inverse_language_data.sh --status

bash experiments/so101_stack_two_cubes/scripts/record_inverse_language_data.sh
```

Repeat until 30 successful demonstrations have been retained. Failed or
interrupted demonstrations should not silently remain in the training set.
The recorder reports the current count and refuses to create episode 31.

After reviewing all 30 trajectories, audit every first frame and compare the
two tasks' role-aligned position distributions:

```bash
bash experiments/so101_stack_two_cubes/scripts/prepare_inverse_language_audit.sh
```

The moving yellow cube in the original task is compared with the moving red
cube in the inverse task; likewise, the two base-cube distributions are
compared. Every color detection must be clean. For each role and image axis,
the median shift must be at most 0.12 normalized image units and the narrower
task's p10--p90 interval must overlap the other by at least 25%. This is a
guard against gross layout leakage, not proof that language is the only usable
signal; the paired physical prompt evaluation remains necessary.

## 2. Merge and validate the language dataset

Preview the non-destructive merge command:

```bash
bash experiments/so101_stack_two_cubes/scripts/prepare_language_dataset.sh --dry-run
```

After the second dataset is complete, create the balanced merged dataset:

```bash
bash experiments/so101_stack_two_cubes/scripts/prepare_language_dataset.sh
```

The preparation script first copies the exact committed v2 subset into a
separate dataset and then merges it with the inverse task. Source datasets are
not modified. The output is:

```text
GY-William/lerobot_stack_two_orders_language_v2
```

Before copying or merging, the preparation script reruns the committed
role-aligned position-balance gate. The merge preserves the original task index
on every trajectory. The validator rejects a dataset when an episode contains
multiple labels, a task has fewer than 30 episodes, task text differs from the
declared vocabulary, or metadata counts disagree with the parquet data.

## 3. Run a resource smoke test before the long job

Download the declared base-model revision into a stable local directory and
verify its 907 MB weight file plus the pinned SmolVLM2 config, tokenizer, and
processor files against the committed SHA-256 manifest:

```bash
bash experiments/so101_stack_two_cubes/scripts/prepare_smolvla_base.sh
```

The manifest pins `lerobot/smolvla_base` at revision
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`. The launcher uses the verified
local snapshot at `/home/hai/models/lerobot_smolvla_base_c83c316`, so a later
Hub update or a network outage cannot silently change the training
initialization. SmolVLM2 processor/config files are pinned separately at
`/home/hai/models/smolvlm2_500m_processor_7b375e1`. Training constructs the
backbone architecture from those local files with `load_vlm_weights=false`,
then restores the complete verified SmolVLA weights. This avoids a redundant
2.03 GB backbone-weight download. Use `--dry-run` to inspect the downloads and
verification command. Interrupted mirror transfers retain their partial file;
the preparation script retries up to 10 times instead of restarting from zero.
If the files were copied from another machine, a successful hash check causes
the script to skip all network access.

The Jetson's configured Aliyun mirror did not expose `num2words`, so install
the two missing packages from the official PyPI index:

```bash
cd ~/lerobot
python -m pip install --index-url https://pypi.org/simple \
  num2words==0.5.14 peft==0.20.0
```

The Jetson environment was checked on 2026-08-05. The installation added only
`docopt`, `num2words`, and `peft`; its dry-run confirmed that Torch and the CUDA
stack would not be replaced. Imports with `PYTHONNOUSERSITE=1`, CUDA BF16
support, and construction of SmolVLA's rank-16 `LoraConfig` all passed. Exact
versions are captured in
[`jetson_smolvla_environment.txt`](../results/jetson_smolvla_environment.txt). The
training launcher checks these dependencies before loading the model.

SmolVLA PEFT reduces trainable parameters but does not remove the base model's
activation memory. First verify model download, one forward/backward pass,
checkpoint saving, and peak Jetson memory with a separate 20-step run:

```bash
bash experiments/so101_stack_two_cubes/scripts/train_smolvla_peft.sh \
  smolvla_lora_r16_two_orders_smoke 20 1 16 20 bf16
```

Do not reuse this output directory for the real run. If the smoke test is
stable, launch the declared 20,000-step adapter run:

```bash
bash experiments/so101_stack_two_cubes/scripts/train_smolvla_peft.sh \
  smolvla_lora_r16_two_orders_20k 20000 1 16 5000 bf16
```

The script starts from the pinned local `lerobot/smolvla_base` snapshot, infers
the SO-101 feature shapes from the merged dataset, applies LoRA to SmolVLA's default language-expert and
state/action projection targets, uses rank 16, and prevents accidental output
overwrites. Mixed precision is controlled explicitly through Accelerate.

## 4. Text evaluation before voice input

Validate language grounding with typed prompts first. Use the same paired
physical starts for every prompt and log all failures with the shared taxonomy.
A reportable first matrix has 10 trials in each cell (40 total):

| Intended behavior | Prompt condition | Prompt |
| --- | --- | --- |
| yellow on red | exact | `Stack the yellow cube on top of the red cube` |
| yellow on red | paraphrase | `Put the yellow block on the red block` |
| red on yellow | exact | `Stack the red cube on top of the yellow cube` |
| red on yellow | paraphrase | `Put the red block on the yellow block` |

Run one 20-second trial with a condition key; repeat each key 10 times under the
same paired physical starts:

```bash
bash experiments/so101_stack_two_cubes/scripts/run_smolvla_language_trial.sh \
  yellow_exact
bash experiments/so101_stack_two_cubes/scripts/run_smolvla_language_trial.sh \
  yellow_paraphrase
bash experiments/so101_stack_two_cubes/scripts/run_smolvla_language_trial.sh \
  red_exact
bash experiments/so101_stack_two_cubes/scripts/run_smolvla_language_trial.sh \
  red_paraphrase
```

The runner fixes the prompt, adapter checkpoint, evaluation dataset, 20-second
horizon, camera settings, and latency log for each condition. Preview any command
without loading the robot by inserting `--dry-run` before the condition.

After reviewing the saved episode, log what stable behavior was actually visible:

```bash
python experiments/so101_stack_two_cubes/scripts/log_smolvla_language_trial.py \
  --condition yellow_exact \
  --observed-behavior yellow_on_red \
  --completion-time-s 14.2
```

If the model makes a stable stack in the opposite order, record it explicitly:

```bash
python experiments/so101_stack_two_cubes/scripts/log_smolvla_language_trial.py \
  --condition yellow_exact \
  --observed-behavior red_on_yellow \
  --failure-label wrong_color_order
```

The logger derives manipulation success and instruction-following success
separately, rejects inconsistent outcome labels, and refuses to log a trial that
does not have a corresponding recorded episode.

Track progress at any time, then enforce the complete 4 x 10 matrix before
reporting it:

```bash
python experiments/so101_stack_two_cubes/scripts/summarize_smolvla_language_trials.py

python experiments/so101_stack_two_cubes/scripts/summarize_smolvla_language_trials.py \
  --require-complete
```

The runner refuses an eleventh episode for any condition. The summarizer reports
Wilson 95% intervals for both metrics and rejects an incomplete or unbalanced
matrix when `--require-complete` is used.

Report both manipulation success and instruction-following accuracy. A stable
stack in the opposite color order is a language-selection failure, not a task
success.

Only after typed prompts work should speech recognition be added as a separate
front end that converts speech to one of these text instructions. Log the audio
transcript and distinguish ASR errors from policy errors; otherwise the project
cannot tell whether a voice-command failure came from speech recognition or
robot control.
