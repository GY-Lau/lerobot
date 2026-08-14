# ACT physical experiment lessons

## Scope

This note records the practical lessons from the SO-101 two-cube stacking ACT
experiments through 2026-08-06. The task is to grasp the yellow cube, place it
on the red cube, release it, and leave a stable stack. Returning to the start
pose is not part of the success criterion.

The short physical screenings below are diagnostic observations, not precise
success-rate estimates. Final comparisons still require a fixed protocol and
more trials.

## What was observed

### Clean 50-episode baseline

`act_stack_two_cubes_v2_50ep_30k` used 50 reviewed demonstrations, 30,000
optimizer updates, batch size 2, AMP disabled, seed 1000, and no image
augmentation. An exploratory physical screening produced one success in
roughly five attempts. The start-pose check was not strict, so this result is a
baseline observation rather than a reportable estimate.

### Geometric plus photometric augmentation

`act_stack_two_cubes_v2_50ep_aug_60k` used brightness and contrast jitter plus
`RandomAffine` rotation, translation, and scale. Its 60k checkpoint completed
with a lower final logged training loss than the earlier baseline, but all
three diagnostic physical trials failed. The recurring failure was inaccurate
final descent: the gripper contacted or pushed the top of the yellow cube.

The saved 30k intermediate checkpoint from the same augmented run was tested
three times:

1. It grasped off-center, then the cube rolled off during placement.
2. It did not secure the cube.
3. It grasped near the center, but missed the placement center and the cube
   rolled off.

This is 2/3 successful grasps but 0/3 completed stable stacks. The 60k model's
grasp phase was worse than the 30k intermediate, while neither checkpoint
solved precise placement reliably.

## Lessons

### Training loss is not task success

A lower imitation loss only shows that optimization fits the sampled training
targets better. It does not guarantee better closed-loop behavior. Physical
success rate and stage-specific failure labels are the relevant deployment
metrics.

### More updates are not monotonically better

The augmented 60k checkpoint performed worse at grasping than its 30k
intermediate. Longer training can amplify bias, overfit a small dataset, or
reinforce inconsistencies introduced by augmentation. Intermediate checkpoints
must be screened instead of assuming that the final checkpoint is best.

### Spatial augmentation can invalidate action labels

An affine transform moves the cube in image coordinates while retaining the
original robot action. For a classification model that may be useful
invariance; for centimeter-level visuomotor control it can create an incorrect
observation-action pair. This is the leading hypothesis for the loss of
precision, but it is not yet a causal conclusion because augmentation and
training duration originally changed together.

Brightness and contrast jitter do not move objects in the image and are a
safer robustness experiment. Even these transforms must be validated on the
robot rather than assumed beneficial.

### Decompose task failure by phase

Record more than a final success flag. For this task the useful phases are:

1. target detection and approach;
2. centered descent;
3. secure grasp;
4. stable transport;
5. alignment above the red cube;
6. low, centered release;
7. stable stack.

This decomposition showed that the 30k augmented model understood the broad
sequence but lacked grasp and placement precision.

### Change one variable at a time

The next controlled comparison holds the 50 episodes, 30k updates, batch size,
AMP setting, hardware, and evaluation protocol fixed:

| Run | Image augmentation | Seed | Question |
| --- | --- | ---: | --- |
| Existing baseline | none | 1000 | reference |
| New treatment | brightness + contrast only | 1000 | effect of photometric augmentation |
| New repeat | none | 2000 | seed sensitivity of the baseline |

The seed-1000 treatment can be compared directly with the existing seed-1000
baseline. The seed-2000 baseline checks whether the original result was highly
dependent on initialization and sample ordering.

Both follow-up runs were launched in parallel on separate RTX A4500 GPUs on
2026-08-06:

- `act_stack_two_cubes_v2_50ep_photo_30k_seed1000` on GPU 0;
- `act_stack_two_cubes_v2_50ep_noaug_30k_seed2000` on GPU 1.

Each run uses 30,000 updates, batch size 2, AMP disabled, and checkpoints every
5,000 updates. The original no-augmentation seed-1000 model is reused rather
than retrained.

## Follow-up physical screening

The two 30k follow-up checkpoints were screened for three episodes each. These
are fault-localization trials, not final success-rate estimates.

### No augmentation, seed 2000

`act_stack_two_cubes_v2_50ep_noaug_30k_seed2000` behaved as follows:

1. The gripper descended onto and became blocked by the top of the yellow cube.
2. The first grasp attempt again contacted the top of the cube. A second
   attempt secured the cube and completed the placement.
3. The first attempt contacted the cube; a later attempt grasped it, but the
   placement did not succeed.
4. The policy secured the yellow cube, but placed it off-center.
5. The gripper again descended onto and became blocked by the top of the yellow
   cube.

The five-episode screening produced three episodes that eventually secured the
cube and one completed placement. The provisional rates are therefore 3/5 for
grasp acquisition and 1/5 for task completion. The repeated top contact and
off-center releases show that changing the random seed can change the outcome
but does not remove the centered-descent or placement-precision problems.

### Brightness and contrast augmentation, seed 1000

`act_stack_two_cubes_v2_50ep_photo_30k_seed1000` failed to secure the yellow
cube in all three episodes. The approach was consistently offset to the left,
and the gripper closed without being centered on the cube.

Brightness and contrast transforms do not geometrically move image pixels, so
the observed offset should not be described as a direct coordinate shift from
the transform. The transforms changed the training samples and the learned
visual representation, however, and this checkpoint produced a repeatable
directional localization error in the physical screening.

### Updated interpretation

- The photometric-augmentation checkpoint is rejected at screening: it showed
  no benefit and a repeated leftward grasp offset in 3/3 episodes.
- Removing `RandomAffine` improved behavior relative to the augmented 60k
  checkpoint, but augmentation was not the only problem. The no-augmentation
  seed-2000 model still repeatedly contacted the top of the cube.
- Seed sensitivity is real, but the seed-2000 checkpoint finished at only 1/5
  task completion and did not solve the underlying centimeter-level grasp and
  placement precision problem.
- Further blind sweeps over seeds, augmentation, or training steps have low
  expected value. The next data collection should emphasize centered grasps
  and low, centered releases rather than another optimizer-only change.

## Batch-size follow-up

A final bounded ACT diagnostic was trained on the same 50 clean episodes with
no augmentation, seed 1000, batch size 8, and 10,000 optimizer updates. It
processed 80,000 sampled frames, approximately 2.67 passes over the 29,914-frame
dataset. Checkpoints were retained at 2.5k, 5k, 7.5k, and 10k updates so that an
early checkpoint could be selected by physical behavior rather than training
loss alone.

### Batch 8 at 2.5k updates

`act_stack_two_cubes_v2_50ep_noaug_batch8_10k_seed1000` checkpoint `002500`
was screened for five episodes. All five failed in the same way: the gripper
descended onto the top of the yellow cube instead of centering around it.

The 2.5k checkpoint is rejected. Its 0/5 result and identical contact pattern
indicate a systematic grasp-alignment error rather than occasional rollout
noise. Testing should advance to the 5k checkpoint; no additional 2.5k trials
are warranted.

## Expert-trajectory replay diagnostic

Five expert episodes from `GY-William/lerobot_stack_two_cubes_v2` were replayed
open-loop on the physical follower. When the red and yellow cubes were placed
to match the recorded episode's initial scene, the recorded trajectories were
physically executable.

This result substantially reduces the likelihood that the low ACT success rate
is primarily caused by corrupted action labels, incompatible motor calibration,
or an inherently unexecutable follower trajectory. It does not prove that the
observation data provide enough information for a learned policy: replay is
given the expert actions directly and does not need to infer grasp depth or
placement geometry from the camera image.

The remaining leading limitations are therefore:

- incomplete observability from a single moving camera, especially during
  occlusion near grasp and release;
- insufficient visual coverage of the placement variations used at evaluation;
- learned visuomotor precision, rather than physical action executability.

A fixed external camera paired with the existing moving camera is the next
high-value intervention. It should first be tested with a small dual-camera
pilot dataset before committing to another full 50-episode collection.

## Camera-pose-2 single-camera pilot

While the second camera is unavailable, a bounded 20-episode pilot will test
the higher, more centered wrist-camera mounting pose. This is an observability
experiment, not an extension of the existing ACT v2 dataset.

- Dataset: `GY-William/lerobot_stack_two_cubes_pose2_20ep`.
- Keep the task, logical camera key (`front`), 640x480 MJPG stream at 30 FPS,
  20-second horizon, robot calibration, and training recipe unchanged.
- Rigidly fix the camera in pose 2 before episode 0 and do not move it during
  recording, training evaluation, or checkpoint comparison.
- The follower was recalibrated immediately before the pilot. Its same physical
  start orientation reads `wrist_roll=-79.69` under the new numeric zero, so
  only this pilot's start-pose gate overrides the old wrist-roll reference.
- Keep both cubes fully visible at the start and record clean, centered grasps
  followed by low, centered releases.
- Do not merge these episodes with the original 50. A model trained only on
  these 20 episodes isolates the effect of the changed viewpoint.

Twenty episodes are enough for feasibility screening, but not for a strong
success-rate claim. If this pilot removes the repeated top-contact failure,
the same pose can be expanded to a larger dataset or paired with the fixed
external camera when it arrives.

## Dual-camera pilot

The second camera is configured as a fixed global view while the original
camera remains on the wrist. The isolated 20-episode dataset is
`GY-William/lerobot_stack_two_cubes_dualcam_20ep`, with logical inputs
`wrist=/dev/video0` and `front=/dev/video2` at 640x480, 30 FPS, MJPG.

The unavoidable black cable is acceptable only if it is fixed in place and
kept outside the gripper path. Record two retained episodes first and review
both videos through grasp and release before completing the remaining 18.

## Decision rule for the next screening

Use two or three trials per checkpoint for rapid fault screening. Stop early if
a repeatable unsafe contact pattern appears. Only models that pass screening
should receive a larger, fixed-protocol evaluation. Do not present the small
screening sample as a final success rate.
