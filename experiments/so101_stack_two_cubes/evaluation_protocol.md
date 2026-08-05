# Evaluation Protocol

This protocol is shared by all policy and checkpoint comparisons. Do not tune
the setup between policies.

## Fixed conditions

- Lock the camera mount, focus, resolution, and exposure.
- Use the same table, lighting, cubes, robot calibration, and initial arm pose.
- Evaluate at 30 FPS with a 20-second episode horizon.
- Do not count reset time as part of an episode.
- Run at least 30 trials for a reportable result.
- Record every trial, including failures.

## Reference arm pose

The reference pose is the per-joint median of frame zero from all 30 original
demonstrations. Body joints are expressed in degrees; the gripper uses its
normalized 0-100 range.

| Joint | Target | Strict tolerance | Relaxed tolerance |
| --- | ---: | ---: | ---: |
| shoulder_pan | -6.15 | +/- 2.50 | +/- 5.00 |
| shoulder_lift | -103.25 | +/- 2.50 | +/- 5.00 |
| elbow_flex | 97.23 | +/- 0.50 | +/- 4.25 |
| wrist_flex | 53.23 | +/- 4.00 | +/- 16.00 |
| wrist_roll | 1.01 | +/- 8.00 | +/- 13.50 |
| gripper | 2.25 | +/- 1.00 | +/- 18.00 |

The default `relaxed` profile covers the complete start-pose range observed in
the 30 demonstrations. It is the gate for generalization evaluation: starts do
not need to reproduce one exact pose, but must remain inside the training
support. Use `--profile strict` only for tightly controlled reproduction.

For a fair policy comparison, ACT and Diffusion must receive the same paired
start poses (or the same declared pose distribution); every trial does not need
to use the single median pose.

Before a reportable trial, check the follower with:

```bash
python experiments/so101_stack_two_cubes/check_start_pose.py --profile relaxed
```

The checker connects only to the motor bus, reads positions, sends no action,
and preserves the existing torque state. A trial is comparable only if the
checker reports `overall: PASS` before recording starts.

For ACT, the recommended runner combines this pose gate with exactly one
20-second episode:

```bash
bash experiments/so101_stack_two_cubes/run_act_trial.sh \
  eval_act_30k_fixed_20s 030000 false
```

Run the same command again after physically resetting the arm and cubes. The
runner detects the existing local dataset and appends one episode. Using one
episode per invocation avoids an uncontrolled reset phase between trials.

For the matched 10/20/30-episode ACT comparison, use the guarded wrapper rather
than editing model paths by hand:

```bash
bash experiments/so101_stack_two_cubes/run_act_data_efficiency_trial.sh \
  --dry-run 10 false

bash experiments/so101_stack_two_cubes/run_act_data_efficiency_trial.sh \
  10 false
```

Replace `10` with `20` or `30` for the other checkpoints. The wrapper maps the
episode count to a distinct evaluation dataset, verifies the exact 30,000-step
training contract, and then delegates to the same pose-gated 20-second runner.
Use `false` for all three reportable runs so AMP is not another changing
variable. Complete the same 30-trial placement schedule for each checkpoint.

## Success definition

A trial succeeds when all conditions hold:

1. The yellow cube is supported by the top face of the red cube.
2. The gripper is no longer supporting or touching the yellow cube.
3. The stack remains standing for at least three seconds.
4. Success occurs within the 20-second episode horizon.

If the policy completes the stack and later knocks it down before three stable
seconds have elapsed, the trial is a failure.

## Placement regimes

Keep the original, unmarked tabletop visible during every evaluation. Do not
add a printed grid or calibration pattern that was absent from the training
demonstrations: changing the background would introduce a visual domain shift.

Cube positions are measured after the fact from each episode's first camera
frame. Pixel centers are stored as normalized camera coordinates `u=x/width`
and `v=y/height`, so the placement distribution remains auditable without
putting guides in the policy's field of view. Before starting a run, save one
reference frame and do not move the camera, robot base, or surrounding objects.

### Fixed

Reset both cubes by eye to the reference-frame positions. The detected start
coordinates, rather than a visible table marker, determine how closely the
placement matches the reference. This tests behavior under minimal variation.

### Bounded random

Randomize both cube centers within the central 10th-to-90th percentile ranges
measured from the 30 training episode starts. Keep the cubes separated and
fully visible. This is the in-distribution generalization condition.

### Position and orientation random

Use locations near the edges of the demonstrated position ranges and vary cube
yaw by eye. This is a harder held-out-layout condition while retaining the
same natural background.

Run the color-based placement audit with:

```bash
python experiments/so101_stack_two_cubes/analyze_cube_placements.py \
  --dataset-root ~/.cache/huggingface/lerobot/GY-William/lerobot_stack_two_cubes \
  --output experiments/so101_stack_two_cubes/training_start_positions.csv
```

The first 10 trials use the reference layout, the next 10 sample
in-distribution positions, and the last 10 use held-out edge layouts. Preserve
the generated schedule and reuse it unchanged for every policy. The placement
detector is an evaluation instrument only; its output is never given to the
policy.

## Failure taxonomy

Assign exactly one primary failure label to every failed trial:

- `perception_or_target_selection`: moves toward the wrong location or cube.
- `approach_miss`: reaches the target but does not align for grasping.
- `grasp_failure`: closes without securing the yellow cube.
- `drop_in_transit`: grasps, then drops before placement.
- `placement_miss`: transports the cube but misses the red cube.
- `unstable_stack`: places the cube but the stack falls within three seconds.
- `post_success_disturbance`: makes a valid stack, then knocks it down.
- `timeout`: has not completed the task at 20 seconds.
- `safety_stop`: human intervention or hardware/software safety stop.
- `other`: document the reason in the notes field.

## Reporting

Report success count, trial count, success rate, and failure counts. Also report
median and P95 episode control-loop latency when latency instrumentation is
available.

After each trial, append one row. A successful example is:

```bash
python experiments/so101_stack_two_cubes/log_trial.py \
  --run-id eval_act_30k_fixed_20s \
  --placement-regime fixed \
  --yellow-position Y0 \
  --red-position R0 \
  --success \
  --completion-time-s 14.2
```

For a failure, replace `--success` with one taxonomy label, for example:

```bash
--failure-label grasp_failure
```

The trial index is assigned automatically within each run. Summarize all runs
with a Wilson 95% confidence interval using:

```bash
python experiments/so101_stack_two_cubes/summarize_trials.py
```

The ACT runner also appends per-frame measurements to
`outputs/eval_latency/<run-id>.csv`. The first 30 frames of each episode are
marked as warm-up and excluded by default. Summarize total command latency and
separate expensive action-chunk refreshes from cached-action frames with:

```bash
python experiments/so101_stack_two_cubes/summarize_latency.py \
  outputs/eval_latency/eval_act_30k_fixed_20s.csv
```
