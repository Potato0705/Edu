# PACE-AES (Layer 2 for WISE-AES)

This package implements the Layer-2 calibration work described in
`D:/SZTU/Education for AI/PACE_AES_method_skeleton.md`. It is additive to the
existing WISE-AES codebase — `wise_aes.py` is imported as-is and its scoring
path is not modified.

## Layout

```
pace/
├── __init__.py
├── llm_backend.py                      # Phase-2: OpenRouter + local Llama-3.1-8B
├── datasets/
│   ├── __init__.py
│   └── asap.py                         # fold-aware ASAP loader (KFold, random_state=42)
└── experiments/
    ├── __init__.py
    ├── run_rq0_diagnostic.py           # Phase 1 output: error decomposition
    ├── plot_rq0.py                     # Figure 2 renderer
    ├── sanity_local_vs_openrouter.py   # Phase 2 gate: y_raw Pearson ≥ 0.85
    └── build_anchor_cache.py           # Phase 2: offline h(e_low/mid/high) cache
```

## RQ0 diagnostic — Protocol-to-Score Misalignment

### Inputs
Existing `logs/exp_{timestamp}_fold{f}/` directories with:
- `config.yaml` (created by `ExperimentManager`)
- `generations/gen_{NNN}.json` (at a target generation, usually `n_generations` from the config)
- Optionally `final_result.json`

### Commands

Run with explicit fold directories (one per fold, mirrors `eval_5fold.py --dirs` semantics):

```bash
cd D:/Python_main/AIforedu/wise-aes
uv run python -m pace.experiments.run_rq0_diagnostic \
    --gen 25 \
    --dirs logs/exp_YYYYMMDD_HHMMSS_fold0 \
           logs/exp_YYYYMMDD_HHMMSS_fold1 \
           logs/exp_YYYYMMDD_HHMMSS_fold2 \
           logs/exp_YYYYMMDD_HHMMSS_fold3 \
           logs/exp_YYYYMMDD_HHMMSS_fold4
```

Or auto-discover the latest run per (prompt, fold) under `logs/`:

```bash
uv run python -m pace.experiments.run_rq0_diagnostic \
    --gen 25 \
    --logs-root logs \
    --prompts 1 2 3 4 5 6 7 8
```

Artifacts land in `results/rq0/`:

| File | Content |
|------|---------|
| `per_essay_predictions.csv` | Long-form `(prompt_id, fold, essay_id, y_true, y_pred, abs_err, band_true, band_pred, band_err, cross_band, essay_len_words)` |
| `error_decomp.csv` | Per-(prompt, fold) aggregates: counts, cross-band error share, off-by-1 / off-by-2+ band shares, mean band distance, QWK |
| `summary.json` | Overall gate statistics, including `cross_band_error_share` and `gate_pass` (True iff ≥ 0.60). An error is *cross-band* iff `band_pred != band_true` (`band_err >= 1`) — the Protocol-to-Score Misalignment signal. |

### Render Figure 2

```bash
uv run python -m pace.experiments.plot_rq0 \
    --in-dir results/rq0 \
    --out-stem results/rq0/rq0_protocol_score_misalignment
```

Outputs `.png` and `.pdf` versions of the three-panel figure (error histogram,
per-prompt cross-band vs same-band share, band-distance CDF).

## Gate

If `summary.json: cross_band_error_share ≥ 0.60`, proceed to Phase 2 (Bridge
layer, `pace/llm_backend.py`). If `< 0.40`, pause and revisit the
Protocol-to-Score Misalignment narrative per the plan's risk §8.6.

## Phase 2 — Bridge layer (local Llama-3.1-8B hidden states)

### Server prerequisites (no HF network)

Because the GPU server has no outbound HF access, download the model on a
local box with HF credentials and upload the weights:

```bash
# on local Windows / Mac
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir models/Meta-Llama-3.1-8B-Instruct \
    --local-dir-use-symlinks False
# then rsync/scp the directory to the server
```

Upload checklist:

| Bundle | Contents | Size | Target path |
|---|---|---|---|
| A · code | `pace/`, `wise_aes.py`, `configs/llama318_{1,3,4,7,8}.yaml`, `configs/default.yaml`, `pyproject.toml`, `uv.lock` | <25MB | `$PROJECT_DIR/` |
| A · data | `data/raw/training_set_rel3.tsv` | ~10MB | `$PROJECT_DIR/data/raw/` |
| B · Layer-1 logs | 5× fold0 dirs (P1/P3/P4/P7/P8: `exp_20251229_{102646,102704,102712,102736,102912}_fold0`) | ~130MB | `$PROJECT_DIR/logs/` |
| C · model weights | `Meta-Llama-3.1-8B-Instruct/` (safetensors + tokenizer) | ~16GB | `$PROJECT_DIR/models/Meta-Llama-3.1-8B-Instruct/` |

Server env:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OPENROUTER_API_KEY=...           # only needed for the sanity script
```

### 8GB-GPU recipe (4-bit NF4)

Llama-3.1-8B in bf16 needs ~16GB VRAM. On an 8GB GPU, load the model in
4-bit NF4 (bitsandbytes). Add to the env:

```bash
pip install "bitsandbytes>=0.43"
```

Then pass `--load-in-4bit` and (optionally) `--max-new-tokens 128` to the
sanity / anchor scripts. The cache signature includes the quant flag, so
4bit and bf16 caches don't collide. Trade-off: hidden states get a small
amount of quant noise; the sanity gate is relaxed to `pearson ≥ 0.75` in
this mode. Downstream residuals `Δ_k = h(x) - h(e_k)` cancel most of the
shared bias as long as both sides use the same backend signature.

### Layer-1 logs layout

`logs/` ships with five fold0 exp dirs (P1/P3/P4/P7/P8), trimmed to only
what Phase-2 reads:

```
logs/
└── exp_20251229_<HHMMSS>_fold0/
    ├── config.yaml
    ├── final_result.json
    └── generations/gen_001.json … gen_025.json
```

`llm_trace.jsonl` and `console.log` are intentionally excluded (each was
~25MB and unused by RQ0/Phase-2). Total `logs/` footprint < 5MB → easy to
sync across machines.

### Sanity: OpenRouter vs local Llama y_raw

```bash
uv run python -m pace.experiments.sanity_local_vs_openrouter \
    --model-path "$PROJECT_DIR/models/Meta-Llama-3.1-8B-Instruct" \
    --logs-root logs \
    --prompts 1 3 4 7 8 \
    --per-prompt 10 \
    --fold 0 \
    --out-dir results/sanity
```

For 8GB GPUs add `--load-in-4bit --max-new-tokens 128`.

Outputs under `results/sanity/`: `per_essay.csv`, `per_prompt_summary.csv`,
`summary.json` (contains `pearson_overall`, `spearman_overall`, `gate_pass`,
`drift_warnings`). Gate rule: `pearson ≥ 0.85` OR (`pearson ≥ 0.80` AND
`spearman ≥ 0.85`).

### Anchor hidden-state cache

```bash
uv run python -m pace.experiments.build_anchor_cache \
    --model-path "$PROJECT_DIR/models/Meta-Llama-3.1-8B-Instruct" \
    --logs-root logs \
    --prompts 1 2 3 4 5 6 7 8 \
    --folds 0 1 2 3 4 \
    --cache-root cache/pace_anchor_cache \
    --dataset asap
```

For 8GB GPUs add `--load-in-4bit --max-new-tokens 128`.

Each `(dataset, prompt, fold)` produces `{dataset}_p{p}_fold{f}.pt` with
`hidden: Tensor[3, hidden_dim]` on CPU fp32 plus metadata. Cache signature
= MD5(`model_path | dtype | pool`) so swapping weights triggers rebuild.
`manifest.json` summarises every cached entry.

Use `--dry-run` on either script to verify dir layout + champion resolution
without loading the model (useful on the local Windows dev box).

## Notes

- Fold splits reuse `KFold(n_splits=5, shuffle=True, random_state=42)` exactly
  as `eval_5fold.py:58-59` does. This ensures the test indices used for the
  diagnostic match the indices seen during the original WISE-AES run.
- `PromptIndividual.evaluate()` already stores per-essay predictions in
  `self.last_pred_scores`, so no modification of `wise_aes.py` is needed.
- Band edges use 7-bucket equal-frequency quantiles **of the train/val
  scores only**, preventing test-set leakage into the band definition.
