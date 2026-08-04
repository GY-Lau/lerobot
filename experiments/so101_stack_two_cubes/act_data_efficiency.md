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
python experiments/so101_stack_two_cubes/generate_act_subsets.py \
  experiments/so101_stack_two_cubes/training_start_positions.csv
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
bash experiments/so101_stack_two_cubes/train_act_data_efficiency.sh --dry-run \
  10 act_stack_two_cubes_10ep_30k

bash experiments/so101_stack_two_cubes/train_act_data_efficiency.sh \
  10 act_stack_two_cubes_10ep_30k

bash experiments/so101_stack_two_cubes/train_act_data_efficiency.sh \
  20 act_stack_two_cubes_20ep_30k
```

Reuse `act_stack_two_cubes_30k/checkpoints/030000` as the 30-episode point; do
not retrain it under a different configuration and silently combine results.

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
