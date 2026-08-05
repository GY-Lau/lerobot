# Reproducible Robot Learning on SO-101

## Project question

How far can a low-cost SO-101 arm and a Jetson Orin NX go on a vision-based
two-cube stacking task, and what changes when demonstration count, policy
family, and deployment method are controlled explicitly?

This is a work-in-progress evidence report. It separates completed engineering
results from experiments that still require physical trials; a trained
checkpoint is not presented as proof of task success.

## System and task

| Component | Configuration |
| --- | --- |
| Robot | SO-101 follower, 6 joint/action dimensions |
| Compute | NVIDIA Jetson Orin NX 16 GB, JetPack 6.2.2, 25 W |
| Sensor | One fixed front RGB camera, 640 x 480 MJPG at 30 FPS |
| Task | Stack the yellow cube on top of the red cube |
| Dataset | `GY-William/lerobot_stack_two_cubes` |
| Demonstrations | 30 episodes, 17,970 frames, about 20 s each |
| Evaluation horizon | 20 s, 30 FPS, pose-gated physical trials |

The original data is intentionally preserved as a v1 baseline. Its visual
audit found hands in some start frames, colored clutter, broad unmarked cube
placements, and 11/30 start frames with at least one ambiguous or missing
color-based placement detection. These limitations are part of the result,
not silently cleaned after training.

## Evidence matrix

| Workstream | Verified evidence | Current status | Missing gate |
| --- | --- | --- | --- |
| ACT baseline | 30k-step checkpoint; exact config and files pass the checkpoint contract | Training complete | Standardized 30-trial physical evaluation |
| ACT data efficiency | Deterministic nested 10/20/30 subsets; all three matched 30k-step checkpoints pass | Training complete | Same paired physical schedule for every checkpoint |
| Diffusion comparison | Same 30 episodes and 60k sampled-frame budget; 30k checkpoint complete | Training complete, stock Jetson deployment not real time | Matched physical comparison requires a disclosed deployable inference setup |
| Jetson latency | Four Diffusion inference configurations with retained log hashes and refresh/cached timing | Complete for the measured configurations | Optional future asynchronous or smaller-policy experiment |
| SmolVLA PEFT | Two-task protocol, merge validator, LoRA launcher, and isolated Jetson environment check | Infrastructure ready | Record 30 real inverse-task demonstrations, then smoke test and train |
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
The reportable data-efficiency result will be physical success rate with Wilson
95% confidence intervals under the unchanged schedule.

The guarded physical runner prevents a common evaluation error: changing the
run label while accidentally loading the 30-episode checkpoint for every test.
It maps `10`, `20`, or `30` to the correct model, validates the training
contract, checks the follower pose, records one 20-second episode, and stores a
separate latency log.

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

## SmolVLA language-control experiment

The existing dataset has one instruction and one behavior. Relabeling the same
trajectory with multiple prompts would test paraphrase invariance at most; it
would not demonstrate language-controlled behavior selection. The declared
experiment therefore requires two balanced, physically distinct tasks:

1. Stack yellow on red: existing 30 demonstrations.
2. Stack red on yellow: 30 new demonstrations under matched conditions.

Only after the merged dataset passes exact vocabulary, episode-balance, and
per-episode label-integrity checks will a rank-16 SmolVLA LoRA smoke test run.
Typed exact and paraphrased prompts are evaluated before speech recognition is
added. This isolates policy grounding errors from ASR errors.

## Reproduce the verified parts

Verify the ACT checkpoints:

```bash
python experiments/so101_stack_two_cubes/verify_act_checkpoint.py \
  outputs/train/act_stack_two_cubes_10ep_30k/checkpoints/030000 \
  --episode-count=10
```

Preview a physical data-efficiency trial without moving the robot:

```bash
bash experiments/so101_stack_two_cubes/run_act_data_efficiency_trial.sh \
  --dry-run 10 false
```

Recompute one Diffusion latency summary from its raw log:

```bash
python experiments/so101_stack_two_cubes/summarize_latency.py \
  outputs/eval_latency/eval_diffusion_30k_n5_amp_fixed_30s.csv \
  --include-warmup
```

Run the experiment-level tests on the Jetson environment:

```bash
/home/hai/miniconda3/envs/lerobot/bin/python -m unittest discover \
  -s experiments/so101_stack_two_cubes -p 'test_*.py'
```

The current suite has 23 passing tests, covering trial logging, summaries,
placement analysis, subset construction, language-dataset validation,
checkpoint contracts, and model-to-evaluation-run mapping.

## Artifact availability and honest boundaries

| Artifact | Current location | Publication status |
| --- | --- | --- |
| v1 demonstrations | Hugging Face repo ID above and Jetson cache | Dataset repo identified; accessibility should be checked before external release |
| ACT 10/20/30 checkpoints | Jetson `outputs/train/` | Complete locally; Hub model publication pending |
| Diffusion 30k checkpoint | Jetson `outputs/train/` | Complete locally; Hub model publication pending |
| Diffusion raw latency logs | Jetson `outputs/eval_latency/` | Machine-readable summaries and hashes committed; raw logs not yet published |
| SmolVLA adapter | Not created | Blocked on real inverse-task demonstrations |
| Source and protocols | Git branch `jetson-py310` | Version controlled and tested |

Claims not yet supported:

- that 10, 20, or 30 demonstrations has the best physical success rate;
- that ACT outperforms Diffusion in task success;
- that SmolVLA follows language commands;
- that voice input controls the robot;
- that the local model artifacts are reproducible from a public model repo.

## Next evidence gates

1. Run the paired 30-trial physical schedule for ACT 10/20/30 and publish the
   success, failure, and latency summaries.
2. Decide whether the Diffusion comparison uses non-Jetson inference or a
   separately disclosed asynchronous/smaller deployment experiment.
3. Record and audit 30 red-on-yellow demonstrations.
4. Run the SmolVLA LoRA resource smoke test, then the declared full training
   and typed-prompt evaluation.
5. Publish immutable dataset/model revisions and raw evaluation artifacts
   before presenting the project as fully reproducible.
