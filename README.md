# WISE-PACE

WISE-PACE is a research prototype for LLM-based Automated Essay Scoring (AES).
It extends WISE-AES from raw validation-score prompt evolution into hidden-evidence-guided
scoring protocol evolution.

The current project objective is:

```text
Find the best scoring protocol P* = <I*, E*>

I: scoring instruction / evolved rubric
E: reference anchors / static exemplars
M: local target LLM used for both scoring and hidden representation extraction
```

The final success metric is still the raw score produced by the LLM under the evolved
protocol `<I*, E*>`. PACE is used as a protocol-selection and evolution signal, not as the
final test-time post-processing calibrator.

## Current Status

Current implementation status:

- Local model backend is enabled; OpenRouter is no longer required for the main experiments.
- Scoring, reflection, rewrite, induction, hidden representation extraction, and PACE fitness all run through the local Llama backend.
- PACE is used during evolution to evaluate candidate protocols with hidden evidence, anchor geometry, stability, and cost signals.
- Final PACE-calibrated inference is disabled by default, because the research target is better raw LLM scoring under `<I*, E*>`, not a stronger post-hoc calibrator.
- Anchor selection and mutation are score-band based and designed to generalize across ASAP prompts P1-P8, not only P1.
- Training curves, generation snapshots, anchor mutation logs, and final candidate comparisons are saved for diagnosis.

Latest single full-fold run:

```text
Config: configs/phase4_full_fold.yaml
Prompt / fold: ASAP P1 fold 0
Model: Meta-Llama-3.1-8B-Instruct local bf16
Runtime: 865.6 minutes on RTX 5090 32GB
Validation primary QWK: 0.5219
Raw test QWK: 0.4780
Raw test MAE: 1.3445
Final PACE tokens: 0
```

Main diagnosis from this run:

- The best raw validation protocol appeared in generation 1 and survived later generations.
- PACE/protocol quality improved slightly, but raw validation QWK did not continue rising.
- High-score prediction remains the biggest weakness. On P1 test, score 12 recall was only 1/14.
- The method is running end-to-end, but it is not yet mature enough for expensive full multi-fold experiments without further improving high-score handling and mutation effectiveness.

## Method Logic Chain

WISE-PACE follows this logic:

```text
1. Build a candidate protocol P = <I, E>
2. Use local LLM M to score validation essays under P
3. Compute raw validation QWK and raw distribution diagnostics
4. Extract hidden states from M under the same scoring context
5. Build an evidence vector z from raw score, hidden geometry, anchor relations, uncertainty, and stability signals
6. Train a small PACE probe on calibration items
7. Evaluate top candidate protocols on held-out fitness items
8. Combine raw QWK, guarded PACE signal, anchor geometry, and protocol-quality guards
9. Select elite protocols
10. Mutate I and E using error cases and hidden-evidence diagnostics
11. Repeat for several generations
12. Before test evaluation, select candidate protocols using validation-only signals
13. Evaluate selected candidate(s) on test with raw LLM scoring
```

The key design principle is:

```text
PACE may guide protocol evolution.
PACE must not become the final scoring model.
```

This avoids the failure mode where the project becomes "a strong calibrator after WISE-AES"
instead of "a better way to find I* and E*".

## Protocol Definition

A protocol is:

```text
P = <I, E>
```

where:

- `I` is the scoring instruction, including the evolved rubric, score range contract, and operational calibration rules.
- `E` is a fixed-size set of reference anchors, each represented by an essay and its known score.

The raw scoring function is:

```text
y_raw = M(x | I, E)
```

The target optimization problem is:

```text
P* = argmax_P QWK(y_raw(P), y)
```

PACE changes how candidate protocols are selected and mutated during search. It does not replace
`y_raw` as the final answer.

## Scoring-Context Encoding

Hidden representations are extracted with a shared scoring-context template so target essays and
anchor essays are comparable.

For a target essay:

```text
Scoring Instruction:
<I>

Score Range:
<score_min>-<score_max>

Reference Anchors:
Score: <low_score>
Essay: <e_low>

Score: <mid_score>
Essay: <e_mid>

Score: <high_score>
Essay: <e_high>

Essay to Represent:
<x>

Representation Target:
Encode the essay to be scored.
```

For an anchor essay:

```text
Scoring Instruction:
<I>

Score Range:
<score_min>-<score_max>

Reference Anchors:
Score: <low_score>
Essay: <e_low>

Score: <mid_score>
Essay: <e_mid>

Score: <high_score>
Essay: <e_high>

Essay to Represent:
<e_low>

Known Score:
<low_score>

Representation Target:
Encode this reference essay as a low-score anchor.
```

The current implementation uses mean-pooled last-layer hidden states over the essay span whenever
span localization is available, falling back to robust sequence pooling when necessary.

## Evidence Vector z

The evidence vector `z` is a compact protocol-quality feature vector. It is built from:

- Raw score features: raw predicted score, normalized score, distance to score boundaries.
- Hidden state features: target representation statistics and projected hidden information.
- Anchor relation features: distance or similarity from target essay to low/mid/high anchors.
- Ordinal geometry features: whether the target is closer to anchors whose scores match the raw prediction.
- Score-consistency features: agreement between raw score and anchor-relative evidence.
- Distribution/stability features: collapse ratio, prediction bias, TV distance, high-score recall.
- Optional enhanced evidence features: extra diagnostics used by PACE and mutation feedback.

The exact feature layout lives in:

```text
pace/pace_fitness.py
```

Current design note: `z` is useful enough for selection diagnostics, but it is still a research
component. The next improvement should make `z` more explicitly decomposed into named feature
blocks and log per-block contribution to protocol selection.

## Evolution of I and E

### Instruction I

The instruction evolves through local-LLM reflection and rewrite:

```text
validation errors + evidence diagnostics -> reflection feedback -> rewritten rubric
```

Important guards:

- Score range normalization keeps rubrics on the dataset score scale.
- Rubrics are treated as holistic ordinal scoring instructions, not point-sum checklists.
- The final score must be an integer inside the dataset score range.
- Operational calibration text is appended to reduce score-scale drift.

### Anchors E

Anchor evolution is score-band based:

```text
low anchor
boundary anchor near mid/high score bands
boundary anchor near high score bands
high/top anchor
```

The implementation avoids P1-only rules. Score bands are derived from the current prompt's
`score_min`, `score_max`, and observed train scores.

High-score handling includes:

- Preference for observed top-score anchors when enough examples exist.
- A top-score preservation probability during mutation.
- Extra penalties when high-score recall or max-score recall collapses.

Anchor mutation can use hidden-evidence representative selection:

```text
candidate pool from same score band
-> evidence/diagnostic scoring
-> optional hidden rerank
-> replacement anchor
```

## Selection and Guards

Candidate selection is raw-first and guarded:

```text
raw_val_qwk
raw_adjusted_qwk
PACE fitness on top candidates
anchor geometry
protocol quality
distribution guard
high-score guard
max-score guard
overfit guard
```

The selection policy is intentionally conservative:

- A candidate with strong PACE but weak raw validation performance should not replace the raw-best protocol.
- PACE can provide a small lift when raw validation is competitive.
- Degenerate PACE results fall back to raw scoring.
- Candidate protocols are selected before test evaluation using validation-only signals.

Final evaluation compares candidate families such as:

```text
best_raw_val
best_raw_guarded
best_pace_guarded
best_pareto
```

Duplicate candidates are detected and reported.

## Repository Layout

```text
wise_aes.py                         Main WISE-PACE evolution runner
pace/
  llm_backend.py                    Local Llama backend for generation and hidden states
  pace_fitness.py                   PACE evidence, calibrator probe, fitness, diagnostics
configs/
  default.yaml                      Backward-compatible default config
  phase4_smoke.yaml                 Fast smoke test
  phase4_evidence_mutation.yaml     Stability/pilot experiment config
  phase4_full_fold.yaml             Single full-fold experiment config
logs/                               Local experiment logs, ignored by git
models/                             Local models, ignored by git
data/                               ASAP data, ignored by git
```

## Running Experiments

Smoke test:

```bash
python wise_aes.py --config configs/phase4_smoke.yaml --fold 0
```

Stability pilot:

```bash
python wise_aes.py --config configs/phase4_evidence_mutation.yaml --fold 0
```

Single full fold:

```bash
python wise_aes.py --config configs/phase4_full_fold.yaml --fold 0
```

On a 32GB GPU, current full-fold P1 runtime is about 14-15 hours with the local Llama-3.1-8B backend.

## Current Research Risks

The model is not yet final. Known risks:

- High-score and max-score recall are still weak.
- Mutation often fails to improve beyond the first strong protocol.
- PACE signals can improve protocol quality without improving raw test QWK.
- `z` needs clearer feature-block attribution.
- The current result is only P1 fold 0; P1-P8 and multi-fold evidence are still required.
- Baselines such as WISE-AES raw-only, OPRO-lite, APE-lite, and random/evidence ablations still need to be run systematically.

## Next Work

Highest-priority next steps:

1. Strengthen high-score and max-score anchor coverage without overfitting to P1.
2. Improve mutation so child protocols can reliably outperform parents.
3. Make `z` feature blocks explicit and log their contribution.
4. Run raw-only and evidence-guided ablations under the same compute budget.
5. Expand from P1 fold 0 to P1-P8 smoke/stability checks before full multi-fold training.

