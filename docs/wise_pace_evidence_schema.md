# WISE-PACE Evidence Vector Schema

This document defines the enhanced compact evidence vector `z` used by
`PaceFitnessEvaluator._build_evidence_bundle`.

Default shape with three anchors is 52 dimensions:

```text
z = [
  y_raw features (5),
  anchor-relative hidden features (19),
  reasoning-text features r_s (11),
  objective essay features f_o (11),
  uncertainty features u (6)
]
```

Raw last-layer hidden states are not concatenated into `z`. The calibrator sees
only compact, interpretable features to reduce overfitting on small calibration
splits.

## 1. Raw Score Features

1. `y_raw_norm`: raw LLM score normalized to `[0,1]`.
2. `y_raw_centered`: `2 * y_raw_norm - 1`.
3. `y_raw_band_low`: one-hot indicator for low score band.
4. `y_raw_band_mid`: one-hot indicator for mid score band.
5. `y_raw_band_high`: one-hot indicator for high score band.

## 2. Anchor-Relative Hidden Features

For anchors `low`, `mid`, and `high`, both target and anchor representations are
encoded under the same scoring-context template:

```text
Scoring Instruction: I
Score Range: score_min-score_max
Reference Anchors: E_low, E_mid, E_high
Essay to Represent: target essay or anchor essay
Known Score: present only for anchors
Representation Target: target/anchor encoding instruction
```

The vector uses these anchor-relative features:

6-8. `anchor_cos_low/mid/high`: cosine similarity between target hidden state
and each anchor hidden state.

9-11. `anchor_cos_dist_low/mid/high`: normalized cosine distance `(1-cos)/2`.

12-14. `anchor_l2_norm_low/mid/high`: L2 distance scaled by `sqrt(hidden_dim)`.

15-17. `anchor_prob_low/mid/high`: softmax over anchor cosine similarities.
Temperature is `pace.anchor_softmax_temperature`.

18-20. `anchor_closest_low/mid/high`: one-hot closest anchor by cosine.

21. `anchor_last_first_cos_delta`: high-anchor cosine minus low-anchor cosine.

22. `anchor_top2_cos_margin`: separation between the closest and second-closest
anchors.

23. `anchor_expected_index_norm`: expected anchor index from soft anchor
probabilities, normalized to `[0,1]`.

24. `anchor_raw_expected_gap`: `y_raw_norm - anchor_expected_index_norm`.

## 3. Reasoning Features `r_s`

25. `reasoning_log_words`
26. `reasoning_has_final_score`
27. `reasoning_trait_hit_rate`
28. `reasoning_strength_hit_rate`
29. `reasoning_risk_hit_rate`
30. `reasoning_compared_rate`
31. `reasoning_example_rate`
32. `reasoning_anchor_rate`
33. `reasoning_score_mentions`
34. `reasoning_score_span`
35. `reasoning_score_std`

## 4. Objective Essay Features `f_o`

36. `essay_log_words`
37. `essay_log_sentences`
38. `essay_log_paragraphs`
39. `essay_mean_sentence_words`
40. `essay_std_sentence_words`
41. `essay_mean_paragraph_words`
42. `essay_type_token_ratio`
43. `essay_long_word_rate`
44. `essay_mean_word_length`
45. `essay_punctuation_rate`
46. `essay_digit_rate`

## 5. Uncertainty Features `u`

47. `uncertainty_hedge_rate`
48. `uncertainty_question_rate`
49. `uncertainty_score_mentions`
50. `uncertainty_score_mention_std`
51. `uncertainty_score_first_last_delta`
52. `uncertainty_raw_norm`

## Evolution Usage

`z` is used in three places:

1. PACE calibrator fitness: train a lightweight ordinal calibrator on the
calibration split, then score the fitness split.
2. Diagnostics: classify hidden mismatch, boundary ambiguity, anchor confusion,
reasoning-score contradiction, and raw score collapse.
3. Evolution guidance:
   - `I` mutation receives evidence diagnostics and score-range constraints.
   - `E` mutation preserves low/mid/high strata and ranks replacement anchors
     by score representativeness, length representativeness, and text diversity.
   - When `evolution.anchor_mutation_hidden_rerank=true`, the best cheap
     candidates are reranked by recomputing candidate anchor hidden states under
     the same scoring context and preferring larger anchor separation.
   - `pace.include_raw_elite_buffer` can evaluate a few extra raw-ranked
     candidates with PACE so later elites are less likely to evolve without
     hidden-evidence diagnostics.
   - Protocol selection uses the full-validation raw QWK for the `beta` term
     while keeping the fitness-split raw QWK as a diagnostic field. This avoids
     comparing PACE-evaluated protocols against non-PACE protocols with
     different raw-QWK denominators.
