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

Record and inspect progress one episode at a time:

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

Run both checkpoints with the same v1 comparison contract: 30,000 optimizer
updates, batch size 2, AMP off, and seed 1000.

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
