# ACT Data-Efficiency Protocol

## Question and limitation

This experiment asks how ACT changes when the number of unique demonstrations
is reduced from 30 to 20 or 10 while optimizer updates, batch size, model,
random seed, and evaluation conditions stay fixed.

The v1 demonstrations contain broad, unmarked placements, visible hands, and
colored clutter. The result is therefore a controlled ablation of this dataset,
not a universal sample-efficiency claim. A cleaner v2 collection is required
before making a strong generalization claim.

## Deterministic nested subsets

`training_start_positions.csv` stores marker-free red/yellow detections from
the first frame of all 30 episodes. The detector found yellow in 29/30 starts;
red clutter made 9 red detections ambiguous or implausibly large. Overall, 19
episodes have clean detections for both colors and 11 are flagged.

Generate the committed subset manifest with:

```bash
python experiments/so101_stack_two_cubes/scripts/generate_act_subsets.py \
  experiments/so101_stack_two_cubes/manifests/training_start_positions.csv
```

The algorithm is deterministic and produces nested sets (`10` is contained in
`20`, which is contained in `30`). Within clean and flagged groups it uses a
farthest-point traversal over robustly normalized red/yellow `(u, v)` positions
to retain spatial coverage. Each subset preserves the full dataset's quality
ratio as closely as integer counts allow:

| Unique episodes | Clean | Flagged |
| ---: | ---: | ---: |
| 10 | 6 | 4 |
| 20 | 13 | 7 |
| 30 | 19 | 11 |

The source CSV SHA-256 is embedded in `act_data_subsets.json`; every training
run recomputes and checks the manifest before loading data.

## Matched training budget

The authoritative 30-episode checkpoint config was inspected directly. The
ablation keeps its settings unchanged:

- ACT with ResNet-18 ImageNet initialization and chunk size 100.
- 30,000 optimizer updates, batch size 2 (60,000 sampled frames).
- AdamW preset, learning rate `1e-5`, no scheduler.
- AMP disabled, seed 1000, no image augmentation, `num_workers=0`.
- Checkpoints every 5,000 updates.

Fixing sampled updates isolates unique trajectory diversity. It also means
smaller subsets see more repeated passes through their frames, which must be
disclosed when interpreting the result.

Preview or launch the missing runs:

```bash
bash experiments/so101_stack_two_cubes/scripts/train_act_data_efficiency.sh --dry-run \
  10 act_stack_two_cubes_10ep_30k

bash experiments/so101_stack_two_cubes/scripts/train_act_data_efficiency.sh \
  10 act_stack_two_cubes_10ep_30k

bash experiments/so101_stack_two_cubes/scripts/train_act_data_efficiency.sh \
  20 act_stack_two_cubes_20ep_30k
```

Reuse `act_stack_two_cubes_30k/checkpoints/030000` as the 30-episode point; do
not retrain it under a different configuration and silently combine results.

Before evaluating a completed subset run, verify both artifact completeness and
the training contract:

```bash
python experiments/so101_stack_two_cubes/scripts/verify_act_checkpoint.py \
  outputs/train/act_stack_two_cubes_10ep_30k/checkpoints/030000 \
  --episode-count=10
```

The verifier checks the final training step, exact episode membership, dataset,
batch size, seed, policy type, AMP setting, optimizer learning rate, scheduler,
and required model/processor files. A directory merely existing is not evidence
that the run completed.

## Completed training runs

All three matched-budget checkpoints now pass `verify_act_checkpoint.py`:

| Unique episodes | Steps | Sampled frames | Final logged loss | Wall time | Contract |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 10 | 30,000 | 60,000 | 0.123 | 04:26:39 | PASS |
| 20 | 30,000 | 60,000 | 0.153 | 04:29:55 | PASS |
| 30 | 30,000 | 60,000 | 0.173 | not recorded | PASS |

The 10/20 losses and wall times come from their retained user-systemd journals;
the 30-episode loss was retained from the original interactive training log.
`act_training_runs.csv` records these sources and the matched configuration.
The non-monotonic final minibatch loss is not a performance ranking: it is one
stochastic training measurement, and the smaller subsets repeat their frames
more often. Physical success under the common trial schedule remains the
data-efficiency endpoint.

## Evaluation and reporting

Evaluate all three checkpoints on the exact same paired physical trial
schedule from `evaluation_protocol.md`. Report:

- unique episodes and frames available to the run;
- optimizer updates and sampled-frame budget;
- success count/rate with Wilson 95% confidence interval;
- failure taxonomy counts;
- median/P95 latency and whether AMP was enabled;
- the v1 data-quality limitation above.

Training loss is a diagnostic, not the data-efficiency result. The primary
comparison is physical success under the unchanged 20-second protocol.

## Exploratory physical screening

An initial five-trial screen was completed for each checkpoint on 2026-08-05.
Cube placement was held to the same small natural range and remained within
the camera view; the videos and per-trial labels are retained. This sample is
useful for finding failure modes, but is too small for a reportable ranking.

| Unique episodes | Success | Wilson 95% CI | Observed failures |
| ---: | ---: | ---: | --- |
| 10 | 2/5 (40%) | 11.8%-76.9% | 3 grasp failures |
| 20 | 3/5 (60%) | 23.1%-88.2% | 2 unstable stacks |
| 30 | 0/5 (0%) | 0.0%-43.4% | 2 placement misses; 1 transit drop; 1 grasp failure; 1 unstable stack |

The intervals overlap substantially. The screen therefore does not establish
that 20 episodes outperform 10 or 30, or that additional demonstrations hurt.
The 30-episode checkpoint's mixed failures are instead a concrete prompt to
inspect demonstration quality, placement coverage, and seed sensitivity.

Latency was effectively matched across all three checkpoints. After warmup,
command P50/P95 was about 15.4/16.7 ms, effective control rate was 29.3 FPS,
and the deadline-miss rate was 0.9% for every run. Action-chunk refreshes took
about 100-102 ms at P50, while cached-action policy time was about 11.7 ms.
Thus the observed success differences are not explained by one checkpoint
running slower. Exact machine-readable values are in
`act_physical_screening.csv`.
