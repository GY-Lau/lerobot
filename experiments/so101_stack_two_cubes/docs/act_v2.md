# ACT v2 Improvement Protocol

## Purpose

The retained v1 experiment measures data efficiency on the original, noisy
30-episode dataset. ACT v2 is a separate improvement experiment: collect 50
clean demonstrations, then compare nested 30/50-episode checkpoints without
rewriting the v1 evidence.

## Collection contract

- Dataset: `GY-William/lerobot_stack_two_cubes_v2`
- Task: `Stack the yellow cube on top of the red cube`
- 50 retained successful episodes, 20 seconds each, 30 FPS.
- Same SO-101 calibration, camera, table, cubes, and lighting as v1.
- Both cubes fully visible at the start with small natural position variation.
- One clean grasp attempt; no failed retries inside an expert trajectory.
- Center the yellow cube over the red cube, lower it close to contact, release,
  and retract without disturbing the stack.

For a recording session, use the persistent recorder. It imports LeRobot once,
then prompts before every episode and after every saved episode. Press Enter to
record; after recording, press Enter/`k` to keep, `d` to discard, `q` to keep
and quit, or `x` to discard and quit:

```bash
PYTHONNOUSERSITE=1 /home/hai/miniconda3/envs/lerobot/bin/python \
  experiments/so101_stack_two_cubes/scripts/record_act_v2_session.py
```

The first startup still takes about 25--30 seconds on the Jetson. Later episodes
reuse the loaded modules, although the arm and camera are safely reconnected for
each episode. The original one-shot recorder remains available as a fallback:

```bash
bash experiments/so101_stack_two_cubes/scripts/record_act_v2_data.sh --status
bash experiments/so101_stack_two_cubes/scripts/record_act_v2_data.sh
```

If the most recently saved attempt is not expert-quality, discard it using the
current episode count as an explicit guard. For example, if the failed attempt
made the dataset contain 12 episodes:

```bash
bash experiments/so101_stack_two_cubes/scripts/discard_last_act_v2_episode.sh 12
```

The removal path retains the original dataset as a backup.

If an interrupted first attempt leaves a zero-episode directory containing
only partial metadata, the recorder detects it, preserves it with an
`_incomplete_<timestamp>` suffix, and creates a fresh dataset instead of
incorrectly passing `--resume=true`.

## Audit and nested subsets

After exactly 50 accepted episodes, extract first-frame cube positions, require
clean red/yellow detections for every episode, and create deterministic nested
30/50 subsets:

```bash
bash experiments/so101_stack_two_cubes/scripts/prepare_act_v2_subsets.sh
```

The 30-episode subset is selected by spatial coverage rather than collection
order. This reduces confounding from operator learning and placement drift over
the recording session.

## Matched training

Run both checkpoints sequentially with the same v1 comparison contract: 30,000
optimizer updates, batch size 2, AMP off, and seed 1000. The sequence verifies
the 30-episode final checkpoint before it starts the 50-episode run. After both
final checkpoints pass, it refreshes `results/model_artifacts.csv` with their
sizes, training contracts, and SHA-256 identities:

```bash
bash experiments/so101_stack_two_cubes/scripts/train_act_v2_sequence.sh
```

The current retained execution is split across machines: v2-30 runs on the
Jetson and v2-50 runs in an isolated RTX A4500 workspace. The dataset manifest,
30,000 updates, batch size, seed, AMP setting, and policy configuration remain
matched. This avoids rerunning v2-50 on the slower Jetson, but machine and
software provenance must be retained and the result must not be presented as a
strict same-hardware training comparison. Both policies are evaluated on the
same Jetson robot setup.

For manual execution or rerunning just one side of the comparison, use:

```bash
bash experiments/so101_stack_two_cubes/scripts/train_act_v2.sh \
  30 act_stack_two_cubes_v2_30ep_30k

bash experiments/so101_stack_two_cubes/scripts/train_act_v2.sh \
  50 act_stack_two_cubes_v2_50ep_30k
```

This separates two questions:

1. v1-30 versus v2-30 estimates the effect of cleaner demonstrations.
2. v2-30 versus v2-50 estimates the effect of more unique clean data.

Do not choose a final policy using training loss alone. Use the same 20-second
physical protocol and retained videos. Start with a small diagnostic screen;
expand the selected v2 model to a reportable evaluation before claiming a
reliable success rate.
