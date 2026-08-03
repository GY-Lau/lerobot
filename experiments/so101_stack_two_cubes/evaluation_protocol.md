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

| Joint | Target | Tolerance |
| --- | ---: | ---: |
| shoulder_pan | -6.15 | +/- 2.50 |
| shoulder_lift | -103.25 | +/- 2.50 |
| elbow_flex | 97.23 | +/- 0.50 |
| wrist_flex | 53.23 | +/- 4.00 |
| wrist_roll | 1.01 | +/- 8.00 |
| gripper | 2.25 | +/- 1.00 |

Before a reportable trial, check the follower with:

```bash
python experiments/so101_stack_two_cubes/check_start_pose.py
```

The checker connects only to the motor bus, reads positions, sends no action,
and preserves the existing torque state. A trial is comparable only if the
checker reports `overall: PASS` before recording starts.

For ACT, the recommended runner combines this pose gate with exactly one
20-second episode:

```bash
bash experiments/so101_stack_two_cubes/run_act_trial.sh \
  act_30k_fixed_20s 030000 false
```

Run the same command again after physically resetting the arm and cubes. The
runner detects the existing local dataset and appends one episode. Using one
episode per invocation avoids an uncontrolled reset phase between trials.

## Success definition

A trial succeeds when all conditions hold:

1. The yellow cube is supported by the top face of the red cube.
2. The gripper is no longer supporting or touching the yellow cube.
3. The stack remains standing for at least three seconds.
4. Success occurs within the 20-second episode horizon.

If the policy completes the stack and later knocks it down before three stable
seconds have elapsed, the trial is a failure.

## Placement regimes

Mark the workspace so placements can be reproduced.

### Fixed

Use one marked center and orientation for each cube. This tests whether the
policy can reproduce the demonstrated behavior under minimal variation.

### Bounded random

Sample each cube center from a marked rectangular region. Record the sampled
position or grid cell. Keep the two regions and minimum cube separation fixed
for every policy.

### Position and orientation random

Use the same regions as bounded random and also sample cube yaw from a fixed
set of marked angles.

The exact region dimensions, grid cells, minimum separation, and yaw values
must be filled in after measuring the physical workspace.

Print [`workspace_grid_a3.svg`](workspace_grid_a3.svg) in A3 landscape mode at
100% / actual size. Verify the printed calibration bar is exactly 100 mm, place
the edge marked `ARM BASE SIDE` toward the robot base, and do not move the mat
between policies.

After choosing reachable, disjoint yellow and red regions, generate the shared
30-trial schedule. The following cells are only an example and must be replaced
with cells verified on the real setup:

```bash
python experiments/so101_stack_two_cubes/generate_trial_plan.py \
  --fixed-yellow G8 \
  --fixed-red M6 \
  --yellow-cells F7 G7 F8 G8 \
  --red-cells L5 M5 L6 M6 \
  --yaw-values 0 45 90 135 \
  --seed 20260803
```

The first 10 trials are fixed, the next 10 randomize position, and the last 10
randomize position and yaw. Reuse the generated CSV unchanged for every policy.

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
  --run-id act_30k_fixed_20s \
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
