# SmolVLA PEFT Language-Control Plan

## Evidence gate

The current dataset was inspected directly on 2026-08-05:

- `GY-William/lerobot_stack_two_cubes`
- 30 episodes, 17,970 frames, 30 FPS
- exactly one task label: `Stack the yellow cube on top of the red cube`

This dataset can support a single-task SmolVLA PEFT smoke test, but it cannot
demonstrate language-conditioned task selection. Relabeling the same trajectory
with paraphrases would teach multiple strings for the same action, not distinct
language-controlled behaviors.

The reportable language experiment therefore uses two balanced behaviors:

1. `Stack the yellow cube on top of the red cube` (existing 30 episodes).
2. `Stack the red cube on top of the yellow cube` (30 new episodes).

Keep camera, background, arm calibration, 30 FPS, 20-second horizon, and cube
placement distribution consistent across both directions. Spread each task
over the same position variations; do not record all examples of one task in
one layout and the other task in another layout, because color order would be
confounded with position.

## 1. Record the inverse task

Each invocation records one auditable episode and resumes the same dataset:

```bash
bash experiments/so101_stack_two_cubes/scripts/record_inverse_language_data.sh --status

bash experiments/so101_stack_two_cubes/scripts/record_inverse_language_data.sh
```

Repeat until 30 successful demonstrations have been retained. Failed or
interrupted demonstrations should not silently remain in the training set.
The recorder reports the current count and refuses to create episode 31. Since
the Jetson is headless, judge each attempt immediately and stop before the next
one if the saved episode needs to be reviewed or removed.

## 2. Merge and validate the language dataset

Preview the non-destructive merge command:

```bash
bash experiments/so101_stack_two_cubes/scripts/prepare_language_dataset.sh --dry-run
```

After the second dataset is complete, create the balanced merged dataset:

```bash
bash experiments/so101_stack_two_cubes/scripts/prepare_language_dataset.sh
```

The output is:

```text
GY-William/lerobot_stack_two_orders_language
```

The merge preserves the original task index on every trajectory. The validator
rejects a dataset when an episode contains multiple labels, a task has fewer
than 30 episodes, task text differs from the declared vocabulary, or metadata
counts disagree with the parquet data.

## 3. Run a resource smoke test before the long job

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

The script starts from `lerobot/smolvla_base`, infers the SO-101 feature shapes
from the merged dataset, applies LoRA to SmolVLA's default language-expert and
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

Report both manipulation success and instruction-following accuracy. A stable
stack in the opposite color order is a language-selection failure, not a task
success.

Only after typed prompts work should speech recognition be added as a separate
front end that converts speech to one of these text instructions. Log the audio
transcript and distinguish ASR errors from policy errors; otherwise the project
cannot tell whether a voice-command failure came from speech recognition or
robot control.
