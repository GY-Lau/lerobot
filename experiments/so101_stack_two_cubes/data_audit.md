# Dataset Audit: v1 Demonstrations

Dataset: `GY-William/lerobot_stack_two_cubes`

This audit documents the original 30 demonstrations before collecting a cleaner
v2 dataset. The intent is to preserve the real baseline rather than silently
changing its conditions.

## Structural checks

| Check | Result |
| --- | --- |
| Episodes | 30 |
| Frames | 17,970 |
| FPS | 30 |
| Mean frames per episode | 599 |
| Mean duration per episode | 19.97 s |
| Camera stream | 640 x 480 RGB |
| Task label | Stack the yellow cube on top of the red cube |

## Initial joint-state audit

The reference below is computed from frame zero of every episode.

| Joint | Median | P10 | P90 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| shoulder_pan | -6.15 | -8.49 | -3.77 | -9.49 | -1.41 |
| shoulder_lift | -103.25 | -103.60 | -101.00 | -103.87 | -98.77 |
| elbow_flex | 97.23 | 97.05 | 97.33 | 93.10 | 97.85 |
| wrist_flex | 53.23 | 49.67 | 54.40 | 46.64 | 68.97 |
| wrist_roll | 1.01 | -6.58 | 5.50 | -12.44 | 9.01 |
| gripper | 2.25 | 2.25 | 2.73 | 2.12 | 20.04 |

Notable start-state outliers include:

- Episodes 0 and 2: wrist-flex values far above the central range.
- Episode 14: wrist-flex value below the central range.
- Episodes 16 and 17: gripper values of 9.59 and 20.04 instead of about 2.25.
- Episode 15: wrist-roll value of -12.44.

The median pose is therefore used as the evaluation reference instead of the
mean. The central 80% range informs the tolerances in
`check_start_pose.py`.

## Visual start-frame audit

A 5 x 6 contact sheet of the 30 episode-start frames was inspected. It can be
regenerated on the Jetson with:

```bash
VIDEO="$HOME/.cache/huggingface/lerobot/GY-William/lerobot_stack_two_cubes/videos/observation.images.front/chunk-000/file-000.mp4"

ffmpeg -y -i "$VIDEO" \
  -vf "select=not(mod(n\,599)),scale=320:240,tile=5x6" \
  -frames:v 1 /tmp/so101_episode_starts.png
```

Observed issues:

- At least two episode starts contain a visible human hand.
- Cube positions span a broad, unmarked area rather than a reproducible grid.
- One or both cubes are close to an image boundary in several starts.
- Some starts place the cubes very close together.
- Persistent clutter includes a dark cable and red/orange tools or containers,
  which can create distractors near the cube colors.

## Consequences

The v1 dataset is sufficient to establish that ACT can learn the task, but it
does not support a clean claim about data efficiency or generalization. The
exploratory 2/10 result mixes model error with inconsistent initial conditions.

## Requirements for v2 collection

1. Fix and mark the camera pose and workspace.
2. Remove unrelated colored clutter from the camera view.
3. Require `check_start_pose.py` to pass before every episode.
4. Use predefined cube grid cells and yaw values.
5. Start recording only after both hands leave the image.
6. Preserve every failed evaluation and assign a failure taxonomy label.

