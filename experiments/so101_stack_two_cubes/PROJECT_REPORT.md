# Reproducible Robot Learning on SO-101

## Project question

How far can a low-cost SO-101 arm and a Jetson Orin NX go on a vision-based
two-cube stacking task, and what changes when demonstration count, policy
family, and deployment method are controlled explicitly?

This is a work-in-progress evidence report. It separates completed engineering
results from experiments that still require physical trials; a trained
checkpoint is not presented as proof of task success.

## Findings

Eight results, each with the number that supports it and the limit that
qualifies it. The rest of this report is the evidence behind them.

**1. Observability is a property of the whole rig, not of the camera.** The
recurring failure was never the policy forgetting the task; it was missing
centimetre precision at exactly two moments, final descent and release. Camera
pose, **gripper orientation**, and **object layout** each changed what the policy
could see at those moments. Re-orienting the gripper for a second camera turned
the wrist view top-down and scored **0/6**; standing it vertical and moving the
red cube laterally so it stays visible during carry reached **2/5**, the project's
best. Two of the three fixes changed no model code at all.

**2. Open-loop fitting error cannot rank policies for closed-loop deployment.**
A teacher-forced diagnostic correctly localised the error to grasp and release,
which is what it is good for. It then ranked wrist-only above wrist+front on
every metric — and the physically stronger policy was wrist+front, because it
**recovers**: bump the cube, lift, re-align, grasp on the second attempt (4 of 5
trials). Teacher forcing scores agreement with an expert who never fails, so it
cannot see recovery. The checkpoint that fit demonstrations best (1.293 deg)
scored 0/6. The sharpest case is a single checkpoint measured against both of
its training layouts: **1.585 deg against 1.304, 22% apart, and 0/5 against
2/5**.

**3. A deployment parameter dominated every model change attempted.** At the
inherited `n_action_steps=20` both vertical checkpoints scored 0 — not imprecise,
**stalled**, re-executing the opening fragment of each fresh 100-step chunk and
never reaching its descend-and-close tail. Setting it to 100, same weights,
moved grasping from **0/5 to 4/5**. For weeks a deployment artifact was
indistinguishable from "the policy cannot grasp".

**4. More data is not automatically better; merges need a distribution check.**
Adding 20 episodes whose red-cube positions did not overlap the evaluation layout
moved the policy from **2/5 to 0/6**, and it failed in a new way: releasing short
of the cube rather than at its edge. The mechanism is measured twice
independently — zero overlap on the red cube's image v-axis predicts a release
21 px short, and a release-phase **bias ratio of 0.92** (against 0.09 in the
control) rules out underfitting in favour of a systematic directional offset.
Caveat: 2/5 versus 0/6 is Fisher p ≈ 0.18, so the claim rests on the two
mechanism measurements agreeing, not on the trial counts.

**5. ACT's CVAE is posterior-collapsed here; it is a plain regressor.** The
appealing explanation for #4 — the latent absorbed the target choice and
inference-time zeroing discards it — was testable, so it was tested. Sampling z
from the prior 32 times beats `z = 0` by **0.6%**, and per-joint spread across
draws is **under 0.04 deg** against errors of 2 to 7 deg. At `kl_weight = 10.0`
on 20-40 episodes the latent carries nothing. So the averaging happens in the
decoder, which never learned to read the target from pixels — leaving exactly two
levers: make the target observable, or use a policy class that **samples** from
the action distribution instead of returning its mean.

**6. Evaluation conditions are part of the experiment.** The demonstrations were
recorded at night with the room light on. Three trials run the next morning with
the light off completed **0/3** grasps; lit, the same checkpoint completed
**5/5**. That is the largest single-factor effect measured anywhere in this
project, and it came from a variable nobody was controlling. The pre-trial gates
check that both cubes are visible; they do not check illumination.

**7. There is no controller that is neutral between these two policies.**
Running each family under both controllers gives opposite answers:

| | synchronous | asynchronous |
| --- | ---: | ---: |
| ACT | 2/8 (25%) | 1/5 (20%) |
| SmolVLA | 0/5 (0%) | **3/5 (60%)** |

SmolVLA needs asynchrony because its 1.255 s inference against 1.67 s of chunk
playback freezes the arm 43% of the time, and **the demonstrations contain no
pauses**; synchronously it failed at approach every time. ACT does not benefit
and its failures changed character: three of five async trials **stalled outright**
— grasped and never lifted, stuttered at the start pose, could not carry the cube
across — which never happened in eight synchronous trials. Its 15 ms inference
lets the server emit a chunk almost every tick, each blended with the last, which
is the frequent-replanning-plus-averaging condition that already made this policy
stutter under temporal ensembling (finding #3). **The best controller depends on
the ratio of inference time to chunk duration, and that ratio differs by 167x
between the two families** (0.753 against 0.0045; the inference times alone
differ by 84x, which is a different number and not the one that matters here), so controller and policy class cannot be
disentangled by picking one for both. An `n_action_steps=25` control also failed
for SmolVLA, ruling out replanning frequency there and leaving continuity.

**8. Sampling removed the averaging; it did not make the second layout work.**
On the merged `combined-40` that cost ACT everything (0/6, releasing short of the
target), SmolVLA scored **2/5**, and none of its failures were that: every error
was at the target or after it, never short of it. But the same checkpoint in the
**other** arrangement it was equally trained on scored **0/5**, with the failures
back at the grasp. That looks like mode collapse and is not: fed expert
observations, the model fits that arrangement almost as well as the one it
succeeds in — **1.585 deg against 1.304, 22% apart** — and this project's own
yardstick is a checkpoint that fit to 1.293 and scored 0/6. **Both modes were
learned; one of them fails in closed loop.** The two joints that set reach and
height, `shoulder_lift` and `elbow_flex`, are the ones ~50% worse there, and in
that arrangement the red cube sits between the gripper and the target — so the
evidence points at occlusion during the approach, not at a missing mode. Third
instance in this report of open-loop fitting failing to predict closed-loop
success.

What none of this establishes yet is in
[Claims not yet supported](#claims-not-yet-supported) — every headline here rests
on 5 to 8 physical trials with overlapping intervals.

## System and task

| Component | Configuration |
| --- | --- |
| Robot | SO-101 follower, 6 joint/action dimensions |
| Compute | NVIDIA Jetson Orin NX 16 GB, JetPack 6.2.2, 25 W (training on 2 x RTX A4500) |
| Sensor | v1/v2: one fixed front RGB camera. Dual-camera generations: wrist `/dev/video0` + fixed front `/dev/video2`, both 640 x 480 MJPG at 30 FPS |
| Task | Stack the yellow cube on top of the red cube |
| Datasets | v1 `GY-William/lerobot_stack_two_cubes`; curated v2 `..._v2`; `..._dualcam_20ep`; `..._vertical_20ep` (+ `_wristonly` ablation); `..._vertical_redleft_20ep`; merged `..._vertical_combined_40ep` |
| Demonstrations | v1: 30 episodes / 17,970 frames; v2: 50 episodes / 29,914 frames; each dual-camera generation: 20 episodes / 11,960 frames; about 20 s each |
| Evaluation horizon | 20 s nominal, 30 FPS, pose-gated physical trials; extended to 35-60 s when studying closed-loop recovery |

The original data is intentionally preserved as a v1 baseline. Its visual
audit found hands in some start frames, colored clutter, broad unmarked cube
placements, and 11/30 start frames with at least one ambiguous or missing
color-based placement detection. These limitations are part of the result,
not silently cleaned after training.

### Where the hyperparameters come from

Every ACT run in this report uses LeRobot's ACT defaults, which are the paper's
values. The training scripts pass `--policy.type`, `--policy.device`,
`--policy.use_amp=false` and `--policy.push_to_hub=false` and **override no
policy hyperparameter at all**: learning rate 1e-5 for both the transformer and
the ResNet18 backbone, weight decay 1e-4, `kl_weight` 10.0, `latent_dim` 32,
`chunk_size` and `n_action_steps` 100, dropout 0.1, ImageNet-pretrained backbone.
Reported differences between ACT runs therefore come from data, camera set, or
deployment settings, never from a quietly retuned optimizer.

One deliberate deviation: LeRobot's default is **100,000** optimizer steps and
this project fixes every run at **30,000**, batch size 8, seed 1000, AMP off. At
batch 8 that is about 20 passes over a 20-episode dataset rather than 67. The
choice predates this report — a 60k run raised overfitting concerns — and it is
held constant across every ACT comparison, but it is a departure from the
published recipe and any absolute ACT number here should be read with it in mind.

The SmolVLA comparison is run the same way: the released checkpoint's own config,
no hyperparameter overrides. Its `scheduler_decay_steps` is 30,000 and LeRobot's
default batch size is 8, so the two policies end up matched on optimizer steps,
batch size, seed, and data exposure **without either being bent to fit the
other**. That coincidence is convenient, not engineered.

## Evidence matrix

| Workstream | Verified evidence | Current status | Missing gate |
| --- | --- | --- | --- |
| ACT baseline | 30k-step checkpoint; exact config and files pass the checkpoint contract | Training complete; exploratory physical screen complete | Reportable physical evaluation if a precise rate estimate is needed |
| ACT data efficiency | Deterministic nested 10/20/30 subsets; all three matched 30k-step checkpoints pass | Five-trial screen complete for every checkpoint | Larger paired sample before ranking checkpoints |
| ACT v2 curation | 50 retained episodes; manual video review; 50/50 clean red and yellow start detections; deterministic nested 30/50 manifest; both 30k checkpoints verified | Training complete; v2-30 on Jetson and v2-50 on an RTX A4500; v2-50 hashes reverified after transfer | Run paired physical evaluation on the same Jetson robot setup |
| Dual-camera + horizontal gripper | 20-episode dataset, 30k checkpoint, teacher-forced diagnostic | 0/6 physical stacks; diagnosed as an observability regression | Superseded by the vertical-gripper generation |
| Teacher-forced diagnostic | Per-phase, per-joint 1-step open-loop action error for four checkpoints | Complete and reusable | None; but see the open-loop/closed-loop caveat below |
| Vertical gripper | Wrist side-view restored, validated in 20/20 recorded episodes; 30k checkpoint | Training and physical trials complete | Larger paired sample |
| Front-camera ablation | wrist+front vs wrist-only on identical data, identical recipe | Both trained and physically tested; open-loop and closed-loop verdicts disagree | Paired trials at the locked deployment config |
| Closed-loop deployment sweep | `n_action_steps` in {1, 20, 50, 100} and temporal ensembling, on hardware | Complete; configuration locked at n=100, ensembling off | None |
| Red-left placement redesign | Occlusion verified fixed in recorded video; 20 episodes collected and merged to 40; both 30k checkpoints trained | Physically tested at n=100: redleft-20 scored 2/5 under matched lighting, the best result in the project so far | Larger paired sample; placement remains the binding failure |
| Layout-mixing / multimodality | Front-camera red-cube distributions (20/20 clean per dataset), signed teacher-forced release error, and a prior-sampling latent probe over 205 release frames | Merging disjoint layouts measurably hurt; underfitting excluded by a 0.92 bias ratio, and the CVAE-latent explanation excluded by the probe | None for the mechanism; the fix now belongs to observability or policy class |
| Illumination sensitivity | Grasp completed in 5/5 lit trials and 0/3 unlit trials of the same checkpoint | Identified from an uncontrolled change during evaluation | Re-run the three unlit trials lit; then either fix lighting in the protocol or record varied-illumination data |
| Diffusion comparison | Same 30 episodes and 60k sampled-frame budget; 30k checkpoint complete | Training complete; not real time under LeRobot's *synchronous* loop, which is the only controller tested | Either an asynchronous controller (chunk N+1 generated while N executes) or a disclosed non-Jetson inference setup |
| Jetson latency | Four Diffusion inference configurations with retained log hashes and refresh/cached timing | Complete for the measured configurations | Optional future asynchronous or smaller-policy experiment |
| SmolVLA vs ACT (single task) | Both arms trained and physically evaluated; 25 trials in `trials.csv`; server-side inference measured at p50 1.255 s | 2x2 complete: SmolVLA 3/5 on red-left and 2/5 on combined against ACT's 2/8 and 0/6, with none of its failures showing ACT's release-short signature | Larger samples |
| The merged checkpoint's swapped-layout failure | Same combined-40 checkpoint in both trained arrangements (2/5 vs 0/5, failures moving from placement to grasp), wrist view verified undegraded, plus a teacher-forced fit against both datasets at 1,500 frames each | Mode collapse **excluded**: fit is 1.585 vs 1.304, and degradation concentrates in `shoulder_lift`/`elbow_flex`. A closed-loop failure, consistent with occlusion during the approach | Direct manipulation of the approach-time occlusion; the mechanism is inferred from where the error sits |
| Controller x policy family | Both families under both controllers, 26 trials: SmolVLA 0/5 sync against 3/5 async, ACT 2/8 sync against 1/5 async with three outright stalls | Complete; the effect is opposite in sign, and the duty-cycle arithmetic explains both directions | None; the confound is irreducible, so each family is reported under the controller that suits it |
| Jetson power mode | MAXN_SUPER with jetson_clocks against the 25 W default, identical weights | Refresh 1,847 ms to 1,070 ms, a 42% reduction | None; it is now held fixed for every SmolVLA trial |
| SmolVLA PEFT (language) | Two-task protocol, pinned base and processor/config manifests, role-aligned layout gate, merge gate, LoRA launcher, four-condition evaluator, and isolated Jetson environment check | Infrastructure ready | Record and audit 30 real inverse-task demonstrations, then smoke test and train |
| Voice control | Text-first evaluation matrix and ASR error-separation design | Designed only | Requires a language-grounded model that first passes typed prompts |

## ACT data-efficiency experiment

The experiment holds architecture, optimizer updates, sampled-frame budget,
batch size, seed, AMP setting, and optimizer constant. Only the number of unique
demonstrations changes. The subsets are nested and quality-stratified using a
deterministic farthest-point traversal over detected red/yellow start
positions.

| Unique episodes | Clean / flagged | Optimizer steps | Sampled frames | Final logged loss | Checkpoint contract |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 10 | 6 / 4 | 30,000 | 60,000 | 0.123 | PASS |
| 20 | 13 / 7 | 30,000 | 60,000 | 0.153 | PASS |
| 30 | 19 / 11 | 30,000 | 60,000 | 0.173 | PASS |

Final minibatch loss is recorded for diagnosis only. It is not monotonic in
unique demonstrations, and smaller subsets repeat their frames more often.
The initial physical screen produced the following results under the unchanged
20-second horizon:

| Unique episodes | Success | Wilson 95% CI | Main observed failures |
| ---: | ---: | ---: | --- |
| 10 | 2/5 (40%) | 11.8%-76.9% | grasp failure (3) |
| 20 | 3/5 (60%) | 23.1%-88.2% | unstable stack (2) |
| 30 | 0/5 (0%) | 0.0%-43.4% | mixed: placement, drop, grasp, stack |

These are exploratory screening results, not reportable success-rate
estimates. The intervals overlap, so this does not establish a best dataset
size or show that more demonstrations reduce performance. It does identify a
non-monotonic result worth investigating with better-curated data and a larger
paired evaluation.

The guarded physical runner prevents a common evaluation error: changing the
run label while accidentally loading the 30-episode checkpoint for every test.
It maps `10`, `20`, or `30` to the correct model, validates the training
contract, checks the follower pose, records one 20-second episode, and stores a
separate latency log.

All three ACT checkpoints ran at essentially the same measured speed after
warmup: command P50/P95 was approximately 15.4/16.7 ms, effective rate was
29.3 FPS, and deadline misses were 0.9%. Their approximately 100 ms action-
chunk refreshes occurred on only 25 of roughly 2,700 analyzed frames per run.
The physical outcome differences therefore are not attributable to different
inference throughput.

## ACT v2 clean-data experiment

The independent v2 collection contains 50 retained episodes and 29,914 frames.
Every trajectory was reviewed as video; six explicitly rejected files were
removed with the pre-edit datasets retained as local backups. The final
start-frame audit found both red and yellow cubes in all 50 episodes with no
ambiguous or implausibly large color component. The detector's size gate is
resolution-relative so a valid nearby cube is not rejected merely for being
wider than a fixed pixel threshold.

The committed manifest deterministically selects a spatially distributed
30-episode subset and nests it inside the complete 50-episode set. Training
holds the v1 contract fixed at 30,000 updates, batch size 2, AMP off, and seed
1000. This creates two controlled comparisons:

1. v1-30 versus v2-30 estimates the effect of cleaner demonstrations.
2. v2-30 versus v2-50 estimates the effect of more unique clean data.

Both final checkpoints pass the contract verifier. They remain hypotheses
until the policies complete the same physical evaluation protocol.

The retained v2-30 run completed on the Jetson in 4:14:51 with final logged
loss 0.153. The v2-50 run completed on an RTX A4500 in 0:39:53 with final
logged loss 0.164, avoiding an unnecessary second long Jetson run. Its final
checkpoint was copied to the Jetson; model and training-config hashes matched
the A4500 originals and the contract verifier passed again. Both runs use the
same dataset manifest and training contract, but this split-hardware execution
is not a strict training-throughput comparison. Physical policy quality is
evaluated on the same Jetson robot setup. A same-GPU replica would be required
before attributing small differences solely to the number of demonstrations.

## Observability experiments: dual camera, gripper orientation, cube layout

This is the project's longest causal chain, and the one that produced its two
most transferable lessons. It starts from the v1/v2 failure signature — the
policy knows the whole sequence but lacks centimeter precision at exactly two
moments, final descent and release — and treats that as a **visual observability**
problem rather than a data, label, calibration, or optimization problem. The
expert-trajectory replay had already eliminated those alternatives: five expert
episodes replayed open-loop on the physical arm executed correctly, so the
actions and hardware are sound.

### Generation 1: dual camera, horizontal gripper — 0/6

A second fixed front camera was added and 20 episodes recorded
(`..._dualcam_20ep`), trained under the fixed contract (30,000 updates, batch 8,
seed 1000, AMP off). Physical result: **0/6 stacks**. Two findings, one
technical and one procedural:

- Technical: re-orienting the gripper for the dual-camera rig turned the wrist
  camera **top-down**. It then saw only the top face of the cube, which carries
  no grasp-height or depth cue, and during placement the **grasped yellow cube
  occluded the red base cube**. Precision therefore fell entirely on one distant,
  shallow-angle fixed camera. The intended fix had introduced a regression in the
  very signal it was meant to improve.
- Procedural: those six outcomes were **never written to any CSV**. The results
  existed only as observations on the Jetson, and the scripts that produced them
  were uncommitted. This report treats that as a result too — an experiment whose
  outcome is not logged is not evidence.

### The measurement that was missing: teacher-forced action error

Until this point the diagnosis was behavioral (failure pattern plus elimination)
with no number attached. `scripts/act/diagnose_teacher_forced_action_error.py` closes
that gap: it feeds the policy the **expert** observations and compares its
predicted action against the expert action frame by frame, aggregated per joint
and per phase, where `d` indexes frames relative to the grasp and release
instants. Units are degrees; lower is closer to the demonstration.

The dual-camera baseline scored **1.293 deg overall**, with the error
concentrated exactly where the physical failures were: **1.897 at grasp (d=0)**
and **2.094 at release (d=-1)**, worst joints shoulder_lift 2.424 and elbow_flex
1.734. This quantitatively confirmed the phase localization.

It also established the report's most important caveat, immediately: the same
checkpoint that fits the demonstrations to 1.29 deg scored **0/6 physically**.
**Low open-loop fitting error does not imply task success.**

### Generation 2: vertical gripper, and a clean front-camera ablation

The gripper was stood **vertical**, restoring the wrist camera's side view. This
was verified before training rather than assumed: a new pre-record gate,
`check_wrist_cube_view.py`, requires both the red and the yellow cube to be fully
in the wrist frame, and it passed in **20/20** recorded episodes
(`..._vertical_20ep`, 20 episodes / 11,960 frames, wrist + front).

Because the front camera's value was still unproven, a wrist-only copy of the
**same** dataset was built by removing `observation.images.front` from
`meta/info.json`, giving a clean single-variable ablation: same episodes, same
actions, same recipe, one input stream removed.

| Metric (deg) | dualcam, horizontal | vertical, wrist+front | vertical, wrist-only |
| --- | ---: | ---: | ---: |
| overall | 1.293 | 1.485 | **1.348** |
| grasp d=0 | 1.897 | 2.053 | **1.675** |
| release d=-1 | 2.094 | 2.237 | **1.424** |
| release d=0 | 1.667 | 2.222 | **1.492** |
| shoulder_lift | 2.424 | 2.570 | **2.334** |
| elbow_flex | 1.734 | 2.230 | **2.105** |
| wrist_roll | 0.788 | 1.083 | **0.851** |

Only the last two columns form a controlled comparison; the dualcam column
crosses a different recording session and is shown for orientation. Wrist-only
fit better on **every** row, and by the widest margin at release (1.424 vs 2.237,
36 percent lower). It also trained about **65 percent faster** (4.0 vs 2.44
steps/s), since one fewer video stream is decoded per sample.

That speed gap turned out to be an artifact worth naming, because it says
nothing about the models. Every training script here inherited
`--num_workers=0`, so video decoding ran inside the training process, serialised
with the forward and backward pass — on a host with 40 cores, one of which was
used. The wrist+front run was not a heavier model, it was decoding one more
stream on the critical path. The training logs show how large that cost was:
`data_s` was **0.275 s against an `updt_s` of 0.185** on the red-left run and
0.375 against 0.143 on the merged one, so **60 to 72 percent of each step was
spent waiting for data**.

A direct A/B settles it. Same dataset, same ACT recipe, same seed, batch 8, 150
steps, one idle GPU:

| | `num_workers=0` | `num_workers=8` |
| --- | ---: | ---: |
| Throughput | 2.51 step/s | **6.68 step/s** |
| `updt_s` (compute) | 0.142 | 0.150 |
| `data_s` (loading) | **0.256** | **0.004** |
| Logged loss at step 50 / 100 | 14.338 / 5.049 | 14.338 / 5.049 |

**2.66x faster, with the compute time unchanged** — the entire gain is `data_s`
falling by a factor of 64. A 30k-step ACT run goes from about 3.3 hours to about
1.25 hours.

The identical logged loss matters as much as the speed: sample order is fixed by
the seed, not by the worker count, so this changes throughput and **not the
training itself**. The setting was initially left at 0 out of a reproducibility
concern that this measurement shows was unfounded.

### Reading `data_s` and `updt_s`, and choosing a worker count

Both come from `lerobot_train.py` and sit back to back in the same loop, so step
wall time is approximately their sum.

`data_s` brackets `next(dl_iter)` plus the preprocessor. What it measures depends
entirely on the worker count: at 0 there are no background processes, so video
decode happens inside that call and is counted; above 0 the workers prefetch and
the call usually returns an already-decoded batch, leaving only preprocessing.
So `data_s` is not "how expensive is data loading" — it is **how long the
training loop sat waiting for data**.

`updt_s` covers `update_policy` end to end: forward, backward, gradient clipping,
optimizer step, and scheduler step. It is pure compute and does not move with the
data pipeline. Both are `AverageMeter`s, so a logged value is the mean since the
previous log line, not an instantaneous reading.

That gives one ratio worth watching, `data_s / (data_s + updt_s)`: the fraction
of each step spent waiting rather than training. Above roughly 30 percent the run
is data-bound and more workers will pay; below about 10 percent it is
compute-bound and they cannot. The ACT runs at `num_workers=0` sat at **60
percent**, which is why the fix was worth 2.66x.

Raising the count past that point does not help, and a sweep on the same dataset
confirms it costs. Measured while an unrelated training job shared the host,
which is the realistic condition here:

| workers | `updt_s` | `data_s` | 150 steps | peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.160 | 0.009 | **39.68 s** | **2.5 GB** |
| 8 | 0.162 | 0.011 | 40.41 s | 3.0 GB |
| 16 | 0.163 | 0.018 | 41.62 s | 4.1 GB |
| 20 | 0.166 | 0.018 | **42.20 s** | **4.6 GB** |

Twenty workers is **6 percent slower than four and uses 84 percent more memory**.
By four, `data_s` is already 5 percent of the step; there is nothing left to
recover, and the extra processes only add scheduling overhead and RSS — `data_s`
itself rises from 0.009 to 0.018.

The count therefore should not be set as a fraction of the core count. It should
be raised until `data_s` is small next to `updt_s` and then left alone: on a
40-core host, 32 cores having nothing to do is the correct outcome for this
workload. The default is half the cores capped at 8 — 8 on the A4500, 4 on the
Jetson, where 4.6 GB of loader memory would also matter on a 16 GB unified-memory
board — and `LEROBOT_NUM_WORKERS` still overrides it. A dataset with more cameras
or higher resolution would move the bound and should be re-measured, not assumed.

Read alone, this says the fixed front camera is a distractor rather than an
information source. The physical trials then showed that reading is wrong, for a
reason worth stating precisely.

### The deployment parameter that dominated everything

At the inherited setting `n_action_steps=20`, both vertical checkpoints scored
**0**: wrist-only 0/5, wrist+front 0/3 (the last two trials were skipped once the
pattern was unambiguous). The failure was not imprecision but **stalling** —
wrist-only hovered over the yellow cube with the gripper open and never closed;
wrist+front never even opened the gripper and jittered in place.

ACT predicts a 100-step (about 3.3 s) chunk covering approach, descent and
closure. Re-planning every 20 steps meant the arm kept re-executing the opening
"approach" fragment of each fresh chunk and never reached the descent-and-close
tail. Raising `n_action_steps` to **100** — same checkpoints, same weights, only
the execution policy changed — moved the grasp rate from 0/5 to **4/5**.

| Config | wrist-only | wrist+front |
| --- | ---: | ---: |
| n=20 | 0/5 | 0/3 |
| n=100 | **1/5** (ep2) | **1/5** (ep0) |

Temporal ensembling, which ACT's authors recommend and which requires
`n_action_steps=1`, was tested at coefficients 0.01, 0.05 and 0.005. All three
**stuttered and could not reach the cube** — worse than n=20. With the horizon
stretched to 120 s it eventually placed accurately, which is diagnostic but not
deployable. An intermediate sweep at n=50 (35 s, 40 s and 60 s horizons) also
hesitated and repeatedly re-adjusted above the cube.

The pattern is monotonic and consistent: **this policy degrades the more often it
re-plans.** Its single-step closed-loop predictions are not stable enough to
drive frequent re-planning, so short chunks produce hesitation loops. The
deployment configuration is therefore locked at **`n_action_steps=100`, temporal
ensembling off**.


### Why the open-loop metric was misleading

At n=100 both variants scored 1/5, but their **failure modes differed
qualitatively**. The wrist+front policy repeatedly **recovered**: it would bump
the top of the yellow cube, lift away, re-align, and grasp on a second or third
attempt (observed in 4 of 5 trials). The wrist-only policy, having missed, rarely
recovered. The front camera's contribution is **closed-loop robustness** — a
global view that lets the policy perceive its own failure and retry.

Teacher-forced error cannot see this. It measures one-step agreement with an
expert who never fails, so it cannot score recovery from self-induced error. It
ranked wrist-only first on every row while the physically more capable policy was
wrist+front. The ablation's numerical verdict was **reversed by direct
observation of behavior**.

The front camera's cost is real too: jerkier motion, visible joint jumps at each
3.3 s re-plan boundary, and recoveries that often exhausted the 20 s horizon.

### Generation 3: red-left layout, to fix placement

With grasping working, the remaining wall is **placement**: cubes released too
high, off-center, or gripped at an edge and dropped. The coarse phase — find the
cube, approach, close — works; what fails is the last centimetre of alignment
before release. That split between a reliable coarse phase and an unreliable
contact-critical one is the general shape of the problem, and it is worth naming
because it predicts which interventions can help: anything that improves gross
trajectory quality will not touch it. This is the original
observability problem displaced to the release instant — at that moment the
**wrist camera is blocked by the held yellow cube** and the **front camera is
blocked by the gripper body**, so nothing sees the red base cube.

The intervention keeps the hardware fixed and changes the **scene**: place the
red cube to the **left** of the yellow one, laterally offset, so the wrist camera
retains a sightline to it while carrying. This was verified before committing to
a full recording session — in the first two episodes the red cube is plainly
visible in the lower wrist frame throughout placement, where previously it was
fully occluded. Twenty consistent episodes were then recorded
(`..._vertical_redleft_20ep`, 20 episodes / 11,960 frames, wrist + front).

Two checkpoints were trained on the A4500 under the unchanged contract, differing
only in data:

| Arm | Dataset | Episodes / frames | Question |
| --- | --- | ---: | --- |
| A | `..._vertical_redleft_20ep` | 20 / 11,960 | Does a consistent, unoccluded placement layout alone fix placement? |
| B | `..._vertical_combined_40ep` | 40 / 23,920 | Does adding the earlier 20 vertical episodes help, or does mixing two layouts hurt? |

Both were then evaluated on hardware at the locked deployment configuration
(n=100, ensembling off), with the episode horizon extended to 40 s.

| Checkpoint | Trials | Success | Wilson 95% CI |
| --- | ---: | ---: | --- |
| A, redleft-20, all trials | 8 | 2 (25%) | 7.1% - 59.1% |
| A, redleft-20, matched lighting only | 5 | 2 (40%) | 11.8% - 76.9% |
| B, combined-40, matched lighting | 6 | 0 (0%) | 0.0% - 39.0% |

**2/5 is the best result the project has produced.** Two findings came out of
these fourteen trials, one of them unplanned.

### An unguarded distribution shift: illumination

The demonstrations were recorded at night with the room light on. The first
three arm-A trials ran the next morning in daylight with the light **off**, and
the policy **never completed a grasp in any of them (0/3)**. With the light on it
completed a grasp in **5/5**.

The three unlit trials are reported separately above rather than dropped, and
the matched-lighting rows carry the smaller n that honesty requires (finding #6).

### Adding data made it worse, and the mechanism is measurable

Under matched lighting arm B scored 0/6 against arm A's 2/5, and its failures had
a distinct signature: it grasped normally (5/6) but released **short of the red
cube, without reaching even its edge** — something arm A never did. Arm A's
placement failures were "at the edge, then rolled off"; arm B's were "nowhere
near it".

Front-camera start-frame detections locate the red cube in normalized image
coordinates across both datasets (`scripts/common/analyze_cube_placements.py`,
20/20 clean detections in each):

| Dataset | red u (median, range) | red v (median, range) |
| --- | --- | --- |
| vertical-20 (earlier layout) | 0.519 [0.458, 0.585] | 0.277 [0.248, 0.330] |
| redleft-20 (evaluation layout) | 0.558 [0.492, 0.605] | 0.431 [0.398, 0.453] |

The horizontal axis overlaps substantially. The vertical axis **does not overlap
at all**: 0.330 maximum versus 0.398 minimum, a gap of 0.068 (about 33 px at 480
height), with medians 0.154 apart (about 74 px). A policy splitting the
difference would release near v = 0.354 — **0.044 short of the nearest edge of
the evaluation-layout red cube, about 21 px**, in the direction actually
observed.

Interpolation and underfitting both produce larger error, so a signed diagnostic
is needed to separate them. Both checkpoints were run teacher-forced on the
**same** redleft episodes and the same 41 release events, and the per-frame CSV
was reduced to a **bias ratio**: absolute mean signed error divided by mean
absolute error. Near 1 means a systematic one-directional offset; near 0 means
zero-mean noise.

| Joint at release | combined-40 signed | ratio | redleft-20 signed | ratio |
| --- | ---: | ---: | ---: | ---: |
| shoulder_lift | -2.030 | **0.93** | +1.690 | 0.91 |
| elbow_flex | +2.836 | **0.92** | -1.266 | 0.79 |
| wrist_flex | +1.118 | 0.75 | +0.334 | 0.30 |
| wrist_roll | +0.835 | 0.84 | +0.421 | 0.42 |
| shoulder_pan | +0.375 | 0.71 | +0.151 | 0.25 |
| gripper | +5.047 | 0.77 | -0.401 | 0.09 |

The merged checkpoint carries a directional bias on **all six** joints; the
control carries one on two. Underfitting predicts a *low* ratio; the measured
ratio is 0.92, so underfitting is excluded. On the two joints that set arm
extension the offsets are **opposite in sign** to the control (differences of
-3.72 and +4.10 degrees), and the gripper opens far more than the expert exactly
where the control is pure noise — matching "released before reaching the cube".

Joint sign conventions were not independently verified, so this establishes a
systematic, direction-consistent configuration offset rather than a specific
geometric direction. The geometric direction comes from the image-space
measurement above. The two lines of evidence are independent and agree.

One caveat on scope: arm A's yellow cube starts within a 0.009-wide band of
normalized u, roughly 6 px. Its 5/5 grasp rate was therefore measured on a
nearly fixed start configuration and does not demonstrate spatial
generalization.

### Testing the architectural explanation, and discarding it

ACT is a CVAE. Its style encoder is fed `[cls, robot_state, action_sequence]`
and **no images** (`src/lerobot/policies/act/modeling_act.py`), so the latent z
is positioned to absorb whatever in the demonstrated action the robot state does
not explain — including which of several valid targets the demonstrator chose. At
inference the latent is set to **zero**: `use_vae` is true, but the sampling
branch is gated on `self.training`, so deployment takes the else-branch and the
decoder returns the **conditional mean** of the action distribution given the
observation.

That suggested an appealing story: the latent absorbs the mode during training,
where the loss looks healthy, and zeroing it at inference discards the answer. It
is testable, so it was tested rather than asserted.

`scripts/act/probe_latent_multimodality.py` injects a chosen latent with a
forward pre-hook on the module that consumes it, then compares, against the
expert action at 205 release frames: the deployed `z = 0` prediction, and
best-of-K over K = 32 draws from the prior. If the latent carried the mode, some
draw should recover the expert action markedly better than zero does.

| | combined-40 | redleft-20 |
| --- | ---: | ---: |
| MAE at `z = 0` (deployment) | 2.479 | 1.829 |
| MAE best-of-32 sampled z | 2.465 (**+0.6%**) | 1.810 (**+1.0%**) |
| MAE mean-of-32 sampled z | 2.480 | 1.829 |
| Largest per-joint spread across draws | **0.027** | **0.032** |

**The hypothesis is false.** Drawing a full prior standard deviation of z moves
the predicted action by hundredths of a degree while the error itself is 2 to 7
degrees, and best-of-32 is worth about one percent. The decoder ignores the
latent almost entirely.

The injection is working, not silently failing: a dead hook would give a spread
of exactly zero, and the measured spread is small but non-zero. Mean-of-K
tracking `z = 0` is the second sanity check.

This is **posterior collapse**. At the default `kl_weight = 10.0` the KL term
pushes the approximate posterior onto the prior, the latent carries no
information, and the decoder learns to ignore it. In this configuration ACT is
effectively `use_vae=false` — a plain L1 regressor with an inert 32-dimensional
input.

Two consequences follow. Raising `kl_weight` to force information out of the
latent is pointless, because it has already collapsed; the lever, if one wanted z
to do work, points the other way. And the averaging is **not** an artifact of
inference-time zeroing: it happens in the decoder, which genuinely never learned
to read the release target from pixels and therefore emits one action — the mean
over the training modes — for a given observation.

The same "averaging destroys multimodal actions" principle accounts for three
independent results in this project:

| Level | Result | Evidence |
| --- | --- | --- |
| Data | merging two disjoint target layouts hurt (0/6 vs 2/5) | zero overlap on the red cube's image v-axis; predicted release 21 px short |
| Policy class | the head is unimodal, so one observation yields one action — the mean over the training modes | release bias ratio 0.92 vs 0.09 in the control; latent probe shows the CVAE is inert, so this is a plain regressor |
| Deployment | small `n_action_steps`, and temporal ensembling, average more | 0/5 grasping at n=20 vs 4/5 at n=100 |

Temporal ensembling is the same effect at a third level: it averages overlapping
chunks, and LeRobot requires `n_action_steps=1` when it is enabled — which is
why every ensembling coefficient tested stuttered.

Note what the middle row is **not**. The appealing version of it — the latent
absorbed the mode and inference-time zeroing threw it away — was tested and is
false. The latent is collapsed, so there is no hidden answer to recover and no
latent-side hyperparameter that fixes this. What remains is the plainer
statement: a policy whose head returns a single action per observation cannot
represent two valid targets, so it returns their average.

That leaves exactly two levers. Make the target observable at the moment it
matters, which is what the gripper-orientation and red-left changes did and what
the remaining placement failures still call for. Or use a policy class that
**samples** from the action distribution instead of returning its mean — diffusion,
or the flow matching used by SmolVLA — which removes the failure by construction.
The second is the motivation for the SmolVLA comparison on this same data.

## ACT versus Diffusion on Jetson

ACT predicts a complete action chunk in one forward pass and executes cached
actions between refreshes. The matched 263M-parameter Diffusion Policy performs
multiple denoising passes whenever its action queue is refreshed. In LeRobot's
stock synchronous control loop, the robot therefore pauses during denoising
and then executes cached actions in a burst.

| Diffusion reverse steps | Executed action steps | AMP | Refresh P50 | Effective FPS | Observation |
| ---: | ---: | :---: | ---: | ---: | --- |
| 100 | 8 | no | 21,085.95 ms | 0.047 | First action arrived after the episode horizon |
| 10 | 8 | yes | 1,835.15 ms | 3.408 | Long pause, then action burst |
| 5 | 8 | yes | 973.89 ms | 6.268 | Pause-burst motion |
| 2 | 15 | yes | 466.48 ms | 15.081 | Jerky motion and no grasp |

Every measured chunk refresh missed the 33.33 ms control deadline. The
defensible conclusion is a deployment result, not a universal policy-quality
claim: the trained Diffusion model is not real time on this Jetson with the
current architecture and synchronous controller. Reducing denoising steps
improves latency but changes the policy computation and degraded the observed
motion.

The qualifier "with the current synchronous controller" is doing real work and
should not be dropped. The pause-then-burst pattern above comes from a control
loop that **stops and waits** for the next chunk. Generating the next chunk while
the current one is still executing removes the stall without making the model any
faster, at the cost of having to join a new chunk onto a trajectory already in
motion. Physical Intelligence's real-time chunking (RTC) treats exactly that join
as the problem to solve. **This project has not implemented or evaluated it**, and
the numbers above say nothing about it; the honest statement is that the
measured setup is not real time and that an asynchronous alternative exists and
was not tried, not that diffusion-class policies cannot run on this hardware.

The same qualifier applies to the SmolVLA latency measurement still to come:
whatever it reports will characterise LeRobot's synchronous loop, not the policy's
ceiling.

## SmolVLA versus ACT on identical data

The multimodality result above leaves two levers: make the target observable, or
use a policy class that samples from the action distribution instead of returning
its mean. SmolVLA's flow-matching head is the second lever, and testing it takes
no new data — both datasets already exist. Two questions are easily confused
here, and only one of them is about multimodality:

- **Does a pretrained VLA beat a from-scratch ACT on 20 demonstrations?**
  Answered on `redleft-20`, which sits close to a single target layout.
- **Does sampling instead of averaging survive the merge that broke ACT?**
  Only answerable on `combined-40`, where the two layouts do not overlap. This
  is the dataset ACT scored 0/6 on by releasing between the modes. **Red-left
  cannot test it**, because there is no second mode there to average over.

Both arms ran the same recipe: the released checkpoint's own config, batch 8,
30,000 steps, seed 1000, no hyperparameter overrides — the ACT contract as well.

### No controller is neutral between these two policies

The obvious plan, run both families synchronously as the ACT trials were, turns
out to handicap only one of them. Inference cost relative to chunk playback
decides how much of a trial the arm spends moving:

| Policy | Inference | Chunk playback at 30 fps | Time in motion |
| --- | ---: | ---: | ---: |
| ACT | 15 ms | 3.33 s (100 steps) | **100%** |
| SmolVLA | 1,255 ms | 1.67 s (50 steps) | **57%** |

ACT's synchronous loop is continuous; 15 ms against 3.3 s is nothing. SmolVLA's
freezes for 1.25 s out of every 2.9 — and **the demonstrations contain no
pauses**, so stop-start execution is out of distribution in a dimension nobody
had been tracking.

Running both families under both controllers, 26 trials, gives opposite answers:

| | synchronous | asynchronous |
| --- | ---: | ---: |
| ACT `redleft` | 2/8 (25%) | 1/5 (20%) |
| SmolVLA `redleft` | 0/5 (0%) | **3/5 (60%)** |

For SmolVLA the controller is the difference between failing at approach every
time and completing three of five. An `n_action_steps=25` control reacted faster
and failed the same way: halving the chunk halves the blind stretch but not the
freeze, since inference costs the same either way, and the duty cycle drops from
57% to 40% moving. That rules out replanning frequency and leaves **motion
continuity**.

For ACT asynchrony is not an improvement (p = 1.00) and it introduces a failure
mode that eight synchronous trials never produced: **three of five async trials
stalled outright** — grasped and never lifted, stuttered at the start pose for
the whole horizon, tried to carry the cube across and could not. At 15 ms per
inference the server can emit a fresh chunk on nearly every control tick, and
each one is blended into the queue by `weighted_average`. That is replanning
every step with averaging, which is precisely the condition under which this
policy stuttered during the temporal-ensembling sweep in finding #3.

So the two families want opposite controllers, for the same reason expressed
twice: **the ratio of inference time to chunk duration differs between them by
roughly 167x** -- 1.255 s against 1.67 s of chunk is 0.753, while 15 ms against
3.33 s is 0.0045. The inference times on their own differ by only 84x; it is the
ratio, not the latency, that decides which controller a policy wants. Choosing one controller for both does not remove the confound, it
relocates it. Asynchronous inference is one of the SmolVLA paper's own
contributions, so its asynchronous arm is the policy as designed; ACT's
synchronous loop is already continuous and is its own best case. Each is
reported under the controller that suits it, and the two columns of the 2x2
below are **not interchangeable**.

Two Jetson settings had to be right first. **MAXN_SUPER with `jetson_clocks`
cut refresh from 1,847 ms to 1,070 ms, a 42% reduction on identical weights**,
and `HF_HUB_OFFLINE` stopped a Hub lookup for a config already on disk from
blocking on a host with no route to it. Measured server-side inference across a
real trial then settled at p50 **1.255 s** (p95 1.274, n=31) against 1.667 s of
runway, so 30 fps became reachable and the comparison could hold frequency
fixed against ACT.

### The 2x2

Twenty physical trials, all logged to `results/trials.csv` with failure labels:

| | ACT (synchronous) | SmolVLA (asynchronous) |
| --- | ---: | ---: |
| `redleft-20` (near single layout) | 2/8 (25%) | **3/5 (60%)** |
| `combined-40` (two disjoint layouts) | **0/6 (0%)** | **2/5 (40%)** |

No pair is statistically separated: ACT versus SmolVLA on the merged data is
Fisher p ≈ 0.18, and SmolVLA's own 3/5 against 2/5 is p = 1.00. What the counts
do show is that **the merge that cost ACT everything cost SmolVLA nothing
measurable**.

### The failure modes are the evidence, not the counts

ACT's failure on `combined-40` was specific: it grasped normally (5/6) and then
released **short of the red cube, without reaching even its edge**. That is the
interpolation signature, and the report backs it with zero overlap on the red
cube's image v-axis and a release-phase bias ratio of 0.92.

SmolVLA's three failures on the same data are not that:

| Trial | Failure | Nature of the error |
| ---: | --- | --- |
| 1 | Placed correctly, then failed to lift clear and knocked the cube off | placement was **on target** |
| 3 | Released from too high; the cube bounced off | **height**, not lateral |
| 4 | Placement off, while visibly correcting | near the target, adjusting |

**None of them released short of the target.** Every error is at the target or
in the motion after it. The averaging signature does not appear, which is what
the arm was run to find out.

Two trials also showed something ACT never did on this data: the policy
**re-aligned above the cube before descending** (trial 2) and **corrected while
placing** (trial 4). That is the closed-loop recovery finding #2 identified as
invisible to teacher-forced error, now appearing in the policy class that was
predicted to have it.

### The merged model fails in one of its two layouts, but it learned both

The `combined-40` dataset holds two cube arrangements. Every trial above used one
of them — red cube to one side of the yellow. Running the same checkpoint in the
**other** arrangement, which it saw just as many episodes of, gives:

| Arrangement | Result | Where it failed |
| --- | ---: | --- |
| As evaluated above | 2/5 | placement and retreat |
| Swapped, equally in training | **0/5** | **grasp, four times out of five** |

The swap is real and measured, not a description: wrist-view cube centres moved
about 143 px along the image v axis **in opposite directions**, matching the
separation between the two training layouts. The failures changed character
completely — jitter while retrying, bumping the top of the cube, closing the
gripper on nothing — which is what a policy looks like in a configuration it has
no confident action for, not what imprecision looks like.

The obvious reading is mode collapse: the merged checkpoint learned one of its
two modes and committed to it. **That reading is wrong.** The diagnostic that
separates it from a closed-loop failure was generalised to run on SmolVLA and
run on both datasets — 1,500 frames each, stride 8, the same checkpoint fed the
expert's own observations:

| Dataset (arrangement) | Overall 1-step MAE | At grasp | At release | Physical |
| --- | ---: | ---: | ---: | ---: |
| `vertical_redleft_20ep` (evaluated above) | 1.304 | 2.287 | 1.254 | 2/5 |
| `vertical_20ep` (swapped) | **1.585** | 2.530 | 2.190 | **0/5** |
| Ratio | **1.22x** | 1.11x | 1.75x | — |

Units are degrees, averaged over the six joints. Fed the right observations, the
model produces near-expert actions for the arrangement it fails in — **22% worse
than the one it succeeds in**, against a project yardstick where the dual-camera
checkpoint fit to **1.293** and scored **0/6**. A mode it had not learned would
not fit at all. **Both modes are in the weights; one of them collapses in closed
loop.**

That makes this the third instance of finding #2 in this report, and the
sharpest one, since here the two numbers come from a single checkpoint:

| | Open-loop fit | Physical |
| --- | ---: | ---: |
| Dual-camera (Generation 1) | 1.293 (best measured) | 0/6 |
| Wrist-only vs wrist+front | wrist-only better on every phase | wrist+front stronger |
| `combined-40`, swapped layout | 1.585 vs 1.304 | 0/5 vs 2/5 |

Two occlusion hypotheses were excluded before the diagnostic ran. **The wrist
view is not degraded**: in the swapped arrangement both cubes still show their
side faces, so the grasp-height cue the top-down rig lost is present here. And
the failures never reached the release, so the occlusion that motivated the
red-left layout — held cube blocking the wrist view, gripper body blocking the
front camera — cannot be what stopped them.

The per-joint breakdown points at a third. `shoulder_lift` and `elbow_flex`, the
two joints that set the arm's reach and height, are the ones that degrade in the
swapped arrangement — 2.724 and 2.744 against 1.719 and 1.849, about **50%
worse** — while `shoulder_pan` is actually *better* there. Reach and height are
what a depth cue buys, and in that arrangement the red cube sits between the
gripper and the yellow target during the approach. So the arrangement is
measurably harder even open-loop, in exactly the degrees of freedom an
approach-time occlusion would cost, and closed loop it fails outright. That is
consistent, not proven: nothing here rules out some other closed-loop effect,
and the visual test — occluding the target during approach in the arrangement
that works — has not been run.

### A different bottleneck

Across both SmolVLA arms, ten trials, the failures cluster **after** the cube is
positioned: not lifting clear on the way out, releasing from too high, a gripper
that opened, closed and opened again at release in two red-left trials. Grasping
is largely solved; `post_success_disturbance` earned its first real use.

That is a different problem from ACT's, which was placing accurately at all. It
also suggests the next intervention is about the release and retreat phase
rather than about perception or policy class.

### What this does not establish

- That SmolVLA beats ACT. Both differences are within noise at these sample
  sizes, and the two families ran under different controllers because a shared
  one would have handicapped SmolVLA. Controller and policy are confounded.
- That approach-time occlusion is what breaks the swapped arrangement. Mode
  collapse is excluded — the checkpoint fits that arrangement to within 22% of
  the one it succeeds in — and the degradation sits in the two joints that set
  reach and height, which is what losing a depth cue would cost. But no trial has
  manipulated that occlusion directly, so the mechanism is inferred from where
  the error concentrates, not measured.
- That either family would do better under the other's controller. Both were
  run both ways: asynchrony took SmolVLA from 0/5 to 3/5 and left ACT at 1/5
  against 2/8 while adding stalls it never showed synchronously. The confound is
  irreducible, not unmeasured.
- That the pretrained prior is doing the work. SmolVLA's pretraining is
  SO-100 community data and this is an SO-101, which is close, but the wrist
  camera here is mounted rotated 90 degrees from upright and the two cameras
  occupy the slots pretraining used for a top-down and a wrist view. A probe on
  the base checkpoint showed slot assignment moves its output by about a quarter
  of what changing the scene does, so the prior is being consumed off-nominal by
  an unmeasured amount.
- That any of this transfers off this hardware. Every number here depends on a
  1.255 s inference cost that is a property of an Orin NX in MAXN.

## SmolVLA language-control experiment

The curated ACT v2 dataset has one instruction and one behavior. Relabeling the
same trajectory with multiple prompts would test paraphrase invariance at most;
it would not demonstrate language-controlled behavior selection. The declared
experiment therefore requires two balanced, physically distinct tasks:

1. Stack yellow on red: the committed, spatially balanced 30-episode subset of
   the manually reviewed v2 dataset.
2. Stack red on yellow: 30 new demonstrations under matched conditions.

The merge pipeline copies the exact v2 subset before merging, so language order
is not confounded with the lower-quality original 30-episode dataset. A
role-aligned spatial gate also compares moving-cube and base-cube distributions
across the two tasks to catch gross layout leakage. Only after the merged
dataset passes this gate plus exact vocabulary, episode balance, and
per-episode label integrity will a rank-16 SmolVLA LoRA smoke test run.
Typed exact and paraphrased prompts are evaluated before speech recognition is
added. A guarded four-condition runner and dedicated result logger are ready;
the logger separates stable manipulation from instruction correctness, so an
opposite-order stack cannot be reported as language-following success. This
isolates policy grounding errors from ASR errors.

## Reproduce the verified parts

Verify the ACT checkpoints:

```bash
python experiments/so101_stack_two_cubes/scripts/act/verify_act_checkpoint.py \
  outputs/train/act_stack_two_cubes_10ep_30k/checkpoints/030000 \
  --episode-count=10
```

Preview a physical data-efficiency trial without moving the robot:

```bash
bash experiments/so101_stack_two_cubes/scripts/act/run_act_data_efficiency_trial.sh \
  --dry-run 10 false
```

Measure per-phase, per-joint teacher-forced action error for any checkpoint:

```bash
python experiments/so101_stack_two_cubes/scripts/act/diagnose_teacher_forced_action_error.py \
  outputs/train/act_stack_two_cubes_vertical_20ep_b8_30k_seed1000/checkpoints/030000
```

Check the wrist-camera start-scene gate without recording (both cubes must be
fully in frame):

```bash
python experiments/so101_stack_two_cubes/scripts/common/check_wrist_cube_view.py --camera-index 0
```

Preview a vertical physical trial at the locked deployment configuration:

```bash
bash experiments/so101_stack_two_cubes/scripts/act/run_act_vertical_trial.sh wristfront eval_demo 100
```

Rebuild the combined 40-episode dataset from its two inputs:

```bash
python experiments/so101_stack_two_cubes/scripts/common/merge_two_datasets.py --help
```

Recompute one Diffusion latency summary from its raw log:

```bash
python experiments/so101_stack_two_cubes/scripts/common/summarize_latency.py \
  outputs/eval_latency/eval_diffusion_30k_n5_amp_fixed_30s.csv \
  --include-warmup
```

Run the experiment-level tests on the Jetson environment:

```bash
/home/hai/miniconda3/envs/lerobot/bin/python -m unittest discover \
  -s experiments/so101_stack_two_cubes/tests -p 'test_*.py'
```

The current suite has 84 passing tests, covering directory layout and links,
trial logging, summaries,
placement analysis, subset construction, language-dataset and cross-task
position-balance validation,
checkpoint contracts, pinned SmolVLA base verification,
model-to-evaluation-run mapping, and artifact hashing.

Every completed model weight file and its training configuration have been
hashed directly on the Jetson. The exact values are in
`results/model_artifacts.csv`; abbreviated model hashes are shown here for orientation:

| Artifact | Model bytes | Model SHA-256 prefix |
| --- | ---: | --- |
| ACT 10 episodes / 30k | 206,699,736 | `be726501b854` |
| ACT 20 episodes / 30k | 206,699,736 | `3db523867a63` |
| ACT 30 episodes / 30k | 206,699,736 | `7d035e4a0c4c` |
| Diffusion 30 episodes / 30k | 1,051,838,640 | `39a77bdf9809` |
| ACT v2 30 episodes / 30k | 206,699,736 | `8441f2f4def9` |
| ACT v2 50 episodes / 30k | 206,699,736 | `a6f506029789` |

These hashes identify the current local artifacts; they do not by themselves
make the models public. After Hub upload, downloaded files must reproduce these
hashes before the publication status is changed from `local_only`.

## Artifact availability and honest boundaries

| Artifact | Current location | Publication status |
| --- | --- | --- |
| v1 demonstrations | Hugging Face repo ID above and Jetson cache | Dataset repo identified; accessibility should be checked before external release |
| ACT 10/20/30 checkpoints | Jetson `outputs/train/`; hashes in `results/model_artifacts.csv` | Complete locally; Hub model publication pending |
| Diffusion 30k checkpoint | Jetson `outputs/train/`; hash in `results/model_artifacts.csv` | Complete locally; Hub model publication pending |
| Diffusion raw latency logs | Jetson `outputs/eval_latency/` | Machine-readable summaries and hashes committed; raw logs not yet published |
| ACT physical screen | Five recorded trials per 10/20/30 checkpoint plus trial and latency summaries | Complete as exploratory evidence; not a reportable ranking |
| ACT v2 demonstrations | Jetson cache; 50 episodes / 29,914 frames; committed position audit and subset manifest | Collection and audit complete; Hub publication pending |
| ACT v2 checkpoints | Both final checkpoints on Jetson; v2-50 trained in the isolated A4500 workspace | Training and contract verification complete; Hub publication and physical evaluation pending |
| Dual-camera 20ep dataset + checkpoint | Jetson cache; A4500 `datasets/` and `outputs/train/` | Complete; 0/6 physical outcomes recorded in this report only, never logged to CSV at the time |
| Vertical 20ep dataset + checkpoint | Jetson cache; A4500; wrist-only ablation copy alongside | Complete; both variants physically tested |
| Teacher-forced diagnostic summaries | A4500: four ACT checkpoints, plus `combined-40` SmolVLA against both training layouts | SmolVLA pair committed under `results/teacher_forced/`; the four ACT summaries are reproduced in this report but not committed |
| Vertical physical trials (n=20, n=100, ensembling, n=50) | Recorded eval episodes on the Jetson; outcomes narrated per trial | Reported here; not yet normalized into `trials.csv` |
| Red-left 20ep dataset | Jetson cache; transferred to A4500 and load-verified | Collection complete; Hub publication pending |
| Combined 40ep dataset | A4500 `datasets/..._vertical_combined_40ep`, 40 eps / 23,920 frames | Built by `merge_two_datasets.py`; load-verified |
| Red-left A/B checkpoints | A4500 `outputs/train/act_stack_two_cubes_{redleft_20ep,combined_40ep}_b8_30k_seed1000` | Both trained to 30k and physically evaluated; 14 trials logged to `trials.csv` |
| SmolVLA single-task fine-tune | A4500 `outputs/train/smolvla_expert_redleft_20ep_b8_30k` | Training under the released recipe; pinned base verified by sha256 |
| SmolVLA language adapter | Not created | Blocked on real inverse-task demonstrations |
| Source and protocols | Git branch `jetson-py310` | Version controlled and tested |

### Claims not yet supported


- that 10, 20, or 30 demonstrations has the best reportable physical success rate;
- that ACT outperforms Diffusion in task success;
- that SmolVLA follows language commands;
- that voice input controls the robot;
- that the local model artifacts are reproducible from a public model repo;
- that wrist+front beats wrist-only on success rate. At n=100 both scored 1/5.
  The recovery advantage is a repeated behavioral observation, not yet a
  statistically separated success rate;
- that the vertical gripper beats the horizontal dual-camera rig on success rate.
  It is a large improvement (0/6 to 1/5 with 4/5 grasping) confounded with the
  `n_action_steps` change, which alone accounts for 0/5 to 4/5 grasping;
- that the red-left layout beats the earlier vertical layout on success rate.
  2/5 is the best result so far but rests on five matched-lighting trials;
- that merging the two layouts is harmful *by a statistically separated margin*.
  2/5 versus 0/6 gives a Fisher exact p of about 0.18. The claim rests on the
  agreement of two independent mechanism measurements, not on the trial counts;
- that the averaging happens inside the CVAE latent. This was tested and is
  false: the latent is posterior-collapsed, best-of-32 prior draws beat `z = 0`
  by about one percent, and per-joint spread across draws is under 0.04 degrees.
  ACT here is effectively a plain L1 regressor;
- that ACT's variational objective contributes anything in this configuration.
  It does not, at `kl_weight = 10.0` on 20-40 episodes. Whether a lower KL weight
  would make the latent useful here is untested;
- that the redleft grasp rate generalizes spatially. Its yellow cube starts
  within a 0.009-wide band of normalized image u, roughly 6 pixels, so 5/5
  grasping was measured on a nearly fixed start configuration;
- that the red-left layout improves placement. The occlusion fix is verified in
  recorded video; its effect on success is untested;
- that SmolVLA outperforms ACT. Every pair is within noise at these sample sizes
  (Fisher p 0.18 on the merged data), and the two families necessarily ran under
  different controllers, so controller and policy class are confounded;
- that running both families asynchronously removes the confound. It does not.
  ACT went 2/8 to 1/5 (p = 1.00) and picked up a stalling failure mode it never
  showed synchronously, so asynchrony helps the slow policy and hurts the fast
  one. Each family is reported under the controller that suits it, and the two
  columns are not interchangeable;
- that SmolVLA's pretrained prior is what produced the difference. Its
  pretraining is SO-100 community data and this is an SO-101, but the wrist
  camera is mounted rotated 90 degrees from upright and both cameras sit in slots
  pretraining used for other views. A probe on the base checkpoint moved its
  output by about a quarter of a scene change when the slot assignment changed,
  so the prior is being consumed off-nominal by an unmeasured amount;
- that any latency conclusion here transfers off this hardware. Every number
  depends on a 1.255 s inference cost specific to an Orin NX in MAXN.

## Next evidence gates

Ordered by what would change a conclusion, not by effort.

1. **Get the last physical numbers out of narration.** 55 trials across 11
   `run_id`s are in `trials.csv`, and `summarize_trials.py` reproduces their
   rates and intervals from the file. Two results are still narrated only: the
   **0/6 dualcam** outcomes from Generation 1, and the **`n_action_steps=20`
   arm** of the deployment sweep — which matters, because that sweep's 0/5 to
   4/5 is finding #3's entire evidence. The summariser also groups by `run_id`
   and knows nothing about illumination, so the matched-lighting 2/5 is still
   separated by hand.
2. **Re-run the three unlit red-left trials with the light on**, taking that
   estimate from five matched-lighting trials to eight, and decide whether
   illumination is fixed by protocol or covered by recorded data.
3. **Collect placement-focused red-left episodes with a real spread of start
   positions.** The current set sits within about 6 px of one configuration, so
   its 5/5 grasp rate says nothing about spatial generalisation. Do not merge
   them with the earlier layout.
4. **Test the approach-time occlusion directly.** Mode collapse is excluded —
   the merged checkpoint fits the arrangement it fails in to within 22% of the
   one it succeeds in — so its 0/5 there is closed-loop, and the error
   concentrates in the two joints that set reach and height. The cheap
   manipulation is to place the red cube so it occludes the target during the
   approach **in the arrangement that works**: if that arm drops too, the
   mechanism is confirmed without training anything.
5. **Re-run the front-camera ablation properly paired at n=100**, scored with a
   recovery-aware metric — attempts per grasp, time to first stable grasp —
   rather than teacher-forced error, which finding #2 showed ranks it backwards.

Deferred, and why: ACT v2 30/50 physical comparison and the Diffusion deployment
decision are both waiting on robot time that the items above use better; the
two-order language experiment needs 30 new demonstrations; a lower-`kl_weight`
retrain would be interesting but the latent being inert already settles the
question it was asked. Publishing immutable dataset and model revisions is a
release step, not an evidence step.
