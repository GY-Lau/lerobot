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

The checker reads positions and sends no action. A trial is comparable only if
the checker reports `overall: PASS` before recording starts.

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
