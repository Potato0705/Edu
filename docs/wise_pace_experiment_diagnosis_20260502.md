# WISE-PACE Experiment Diagnosis 2026-05-02

## Experiment Scope

All runs use ASAP set 1, fold 0, local Meta-Llama-3.1-8B-Instruct, `n_train=240`,
`n_val=32`, `n_test=32`, `population_size=4`, and `n_generations=3`.

| Run | Exp Dir | Best Val QWK | Test QWK | Duration | Total Tokens All |
| --- | --- | ---: | ---: | ---: | ---: |
| Phase2 no evidence | `logs/exp_20260502_102424_fold0` | 0.6667 | 0.1100 | 70.8 min | 1,852,656 |
| Phase4 light | `logs/exp_20260502_113515_fold0` | 0.5758 | 0.1100 | 81.0 min | 1,960,224 |
| Phase4 full | `logs/exp_20260502_125619_fold0` | 0.5663 | 0.3382 | 94.6 min | 2,198,684 |

## Main Diagnosis

Phase4 full is the only run that improves test QWK, from 0.1100 to 0.3382.
Its validation combined fitness is lower than Phase2, but its selected protocol
generalizes better. This means the Phase2 selection signal is too easy to
overfit on the small validation split, while Phase4 full's hidden-reranked
anchor mutation gives a more robust protocol.

Phase2's best protocol appears in generation 1 with combined fitness 0.6667,
but final test QWK is only 0.1100. The best Phase2 protocol has full-val raw
QWK 0.2476 while PACE QWK is 0.7851, so the combined score is dominated by a
calibrator/fitness-split signal that does not transfer to raw test scoring.

Phase4 light also fails to improve test QWK. Its evidence slot policy forced
all 10 anchor mutations into slot 1, which created a mutation bottleneck. This
confirmed that evidence-suggested slots should be a probability bias, not a hard
rule. The code has already been changed to support `evidence_anchor_slot_prob`,
and Phase4 full used `0.6`, producing 6 evidence mutations and 4 random
mutations.

## Generation Behavior

### Phase2 No Evidence

| Gen | Best Combined | Best Raw | Raw Best | Hidden Rerank |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6667 | 0.2476 | 0.3248 | 0 |
| 2 | 0.6260 | 0.3001 | 0.3001 | 0 |
| 3 | 0.5948 | 0.3001 | 0.3001 | 0 |

Phase2 finds a high PACE protocol early, but the selected protocol is not the
raw-val best in generation 1. Later generations do not surpass generation 1.

### Phase4 Light

| Gen | Best Combined | Best Raw | Raw Best | Hidden Rerank |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.5758 | 0.2476 | 0.3248 | 0 |
| 2 | 0.5595 | 0.2476 | 0.2476 | 0 |
| 3 | 0.5238 | 0.2476 | 0.2476 | 0 |

All 10 mutations used evidence slot 1. This reduced exploration and repeatedly
generated weaker mid-anchor replacements.

### Phase4 Full

| Gen | Best Combined | Best Raw | Raw Best | Hidden Rerank |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.5646 | 0.2476 | 0.3248 | 3 |
| 2 | 0.5663 | 0.2808 | 0.2808 | 4 |
| 3 | 0.5184 | 0.2808 | 0.2808 | 3 |

Phase4 full selected anchors `[630, 1678, 1386]` and reached test QWK 0.3382.
Hidden rerank was used on all 10 mutations, and mutation sources were balanced:
6 evidence-guided, 4 random.

## Error Diagnosis

Across PACE-evaluated individuals, the dominant evidence problems are:

| Run | Boundary Ambiguity | Reasoning Contradiction | Under-score High Hidden |
| --- | ---: | ---: | ---: |
| Phase2 | 55 | 47 | 25 |
| Phase4 light | 53 | 62 | 7 |
| Phase4 full | 54 | 55 | 8 |

The scorer still systematically under-scores high-quality essays.

| Run | Test MAE | Bias `pred-truth` | Exact | Under | Over | Severe `abs>=3` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase2 | 2.094 | -1.469 | 3 | 24 | 5 | 10 |
| Phase4 light | 2.094 | -1.469 | 3 | 24 | 5 | 10 |
| Phase4 full | 1.875 | -1.438 | 5 | 21 | 6 | 9 |

Phase4 full improves ranking enough for QWK, but absolute calibration remains
weak. Mean predictions by true score in Phase4 full:

| True | Mean Pred |
| ---: | ---: |
| 6 | 5.33 |
| 7 | 5.67 |
| 8 | 6.55 |
| 9 | 7.86 |
| 10 | 8.17 |
| 11 | 8.50 |

High scores are still compressed downward, especially true 10-11 essays.

## Evidence Vector Diagnosis

The 52-dimensional compact evidence vector is useful enough to guide selection
and anchor mutation, but it is not yet mature:

1. The current `z` is mostly local to each essay and anchor context. It lacks
   generation-level stability features such as score variance across equivalent
   prompts, repeated decoding consistency, and raw-score distribution collapse.
2. Anchor-relative geometry helps, but anchor geometry alone can be misleading.
   Phase4 light produced higher or comparable anchor geometry for some mid-anchor
   replacements while raw QWK got worse.
3. The calibrator is trained on only 16 calibration examples and tested on 16
   fitness examples. This explains why PACE QWK can be high while raw test QWK
   remains weak.
4. `z` currently represents hidden relation to anchors, reasoning text, objective
   essay features, and uncertainty, but it does not directly encode ordinal
   distance to score labels beyond the anchor slots. Adding score-conditioned
   class prototypes or per-band centroids should improve robustness.

## I/E Evolution Diagnosis

Instruction evolution currently rewrites based on real validation errors plus
evidence diagnostics. This is directionally correct, but the changes are not yet
controlled enough. The model often produces generic feedback such as "improve
general accuracy", which does not create stable rubric improvements.

Anchor evolution is more important in the current results:

1. Random slot mutation in Phase2 keeps diversity, but has no evidence focus.
2. Hard evidence slot mutation in Phase4 light over-focuses on slot 1 and hurts
   exploration.
3. Phase4 full's probabilistic evidence slot plus hidden rerank is the best
   current design. It improves test QWK, but costs more.

## Cost Diagnosis

PACE tokens dominate total cost:

| Run | Base Tokens | PACE Tokens | PACE Share |
| --- | ---: | ---: | ---: |
| Phase2 | 756,218 | 1,096,438 | 59.2% |
| Phase4 light | 803,568 | 1,156,656 | 59.0% |
| Phase4 full | 833,697 | 1,364,987 | 62.1% |

Phase4 full is more expensive because hidden rerank adds representation passes.
It is currently justified only if the test improvement is reproducible across
more folds.

## Code and Method Issues Found

1. Validation overfitting is the main risk. Current pilot uses `n_val=32`, split
   into 16 calib and 16 fitness examples. This is too small for stable PACE QWK.
2. Final test uses raw scorer only. PACE calibrator is used for protocol
   selection, not final prediction. This creates objective mismatch.
3. Early rejection did not trigger in these runs. Thresholds are too permissive
   or the mini-set is too small to reject poor protocols.
4. Evidence anchor slot should not be deterministic. This has been patched with
   `evidence_anchor_slot_prob`.
5. Hidden rerank is beneficial but expensive. Candidate reuse/cache is needed.
6. Score calibration remains weak. The local Llama scorer compresses high scores
   downward even after protocol evolution.

## Recommended Next Steps

1. Run Phase4 full on at least 3 folds with the same seed policy. Treat current
   test QWK 0.3382 as promising but not mature.
2. Increase validation budget: use at least `n_val=96` or `n_val=128`, with
   larger calib/fitness splits.
3. Add a final PACE-calibrated inference option and report both raw-test QWK and
   PACE-calibrated-test QWK.
4. Replace hard PACE QWK with a stability-aware objective:
   `combined = raw_val + pace_delta + anchor_geometry + stability - cost`.
5. Improve `z` with per-band centroid distances, ordinal margin features, and
   score-distribution stability features.
6. Make instruction evolution structured: output rubric patch operations rather
   than free-form rewrites.
7. Add anchor cache for hidden rerank candidate representations to reduce Phase4
   full cost.
8. Add baselines before claiming maturity: raw WISE-AES, Phase2, Phase4 full,
   OPRO-lite, and APE-lite on the same folds.

## Current Maturity Assessment

The model is not mature yet. The current status is a promising Phase4 pilot:
the full hidden-evidence-guided design improves test QWK in this run, but the
system still has validation overfitting, scorer under-calibration, high PACE
cost, and insufficient fold-level evidence.
