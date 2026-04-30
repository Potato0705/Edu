"""RQ0 diagnostic: Protocol-to-Score Misalignment for WISE-AES.

Inputs
------
- One or more ``logs/exp_{timestamp}_fold{f}`` directories produced by
  ``run_5fold_experiment.py`` (one per fold). Each must contain:
    * ``config.yaml``
    * ``generations/gen_{NNN}.json``  (at the target generation)
    * optionally ``final_result.json``

Outputs
-------
- ``results/rq0/per_essay_predictions.csv`` — long-form per-essay (y_true, y_pred,
  band_true, band_pred, abs_err, band_err, cross_band, essay_len_words, ...).
- ``results/rq0/error_decomp.csv`` — per-(prompt, fold) aggregated error stats
  suitable for Figure 2.
- ``results/rq0/summary.json`` — overall gate statistics, including the
  **cross-band error share** vs. the 60% threshold. An error is *cross-band*
  iff ``band_pred != band_true`` (i.e. ``band_err >= 1``); this is the
  Protocol-to-Score Misalignment signal the paper narrative hinges on.

Usage
-----
    python -m pace.experiments.run_rq0_diagnostic \\
        --gen 25 \\
        --dirs logs/exp_*_fold0 logs/exp_*_fold1 ... \\
        --config-fallback configs/llama318_1.yaml

or, for auto-discovery of the most recent fold-dir per fold::

    python -m pace.experiments.run_rq0_diagnostic \\
        --gen 25 \\
        --logs-root logs \\
        --prompts 1 2 3 4 5 6 7 8

The diagnostic loads each champion (I*, E*) from the requested generation,
reruns scoring on the test fold (the same ``KFold(n_splits=5, shuffle=True,
random_state=42)`` split used by ``eval_5fold.py``), captures per-essay
predictions, and aggregates error statistics.

The script is designed to leave ``wise_aes.py`` untouched. It imports
``PromptIndividual`` and ``SimpleVectorStore`` and reuses the evaluation
pathway exactly as ``eval_5fold.py`` does, but additionally exposes
``last_pred_scores`` for per-essay bookkeeping.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

# Ensure repo root is on sys.path so we can import wise_aes.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import wise_aes  # noqa: E402
from wise_aes import PromptIndividual, SimpleVectorStore  # noqa: E402

from pace.datasets import asap as asap_loader  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Config / dir discovery
# ---------------------------------------------------------------------------


_FOLD_RE = re.compile(r"fold(\d+)")


def _infer_fold(dir_name: str, fallback: int) -> int:
    m = _FOLD_RE.search(dir_name)
    if m:
        return int(m.group(1))
    return fallback


def _discover_latest_per_fold(
    logs_root: Path, prompts: Optional[Iterable[int]] = None
) -> Dict[Tuple[Optional[int], int], Path]:
    """Scan ``logs_root`` and return the most recent ``exp_*_fold{f}`` per fold.

    Key: ``(prompt_or_none, fold)`` where ``prompt_or_none`` may be None if
    prompt can't be inferred from the directory name. We pair with
    ``config.yaml`` later to attach the prompt id.
    """
    out: Dict[Tuple[Optional[int], int], Path] = {}
    for d in sorted(logs_root.glob("exp_*_fold*"), key=lambda p: p.name):
        m = _FOLD_RE.search(d.name)
        if not m:
            continue
        fold = int(m.group(1))
        # Try to parse prompt from path or config
        prompt = None
        cfg_path = d / "config.yaml"
        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                prompt = int(cfg.get("data", {}).get("essay_set"))
            except Exception:
                prompt = None
        if prompts is not None and prompt is not None and prompt not in prompts:
            continue
        # Keep latest (sorted ascending by name; last wins)
        out[(prompt, fold)] = d
    return out


def _find_gen_file(exp_dir: Path, target_gen: int) -> Optional[Path]:
    gen_file = exp_dir / "generations" / f"gen_{target_gen:03d}.json"
    if gen_file.exists():
        return gen_file
    # Fall back to the highest available generation <= target_gen
    gens = sorted(exp_dir.glob("generations/gen_*.json"))
    if not gens:
        return None
    best: Optional[Path] = None
    for p in gens:
        try:
            n = int(p.stem.split("_")[1])
        except Exception:
            continue
        if n <= target_gen and (best is None or int(best.stem.split("_")[1]) < n):
            best = p
    return best or gens[-1]


# ---------------------------------------------------------------------------
# 2. Per-fold rerun
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    prompt: int
    fold: int
    exp_dir: str
    gen: int
    champion_val_qwk: float
    champion_test_qwk: float
    per_essay: pd.DataFrame  # long-form, one row per test essay


def _mock_exp_manager(config: dict) -> None:
    class _Mock:
        def __init__(self, cfg):
            self.config = cfg

        def log_llm_trace(self, _):
            pass

        def track_usage(self, *_):
            pass

        def count_tokens(self, text):
            return max(1, len(text) // 4)

    wise_aes.EXP_MANAGER = _Mock(config)


def _select_champion(gen_snapshot: dict) -> Tuple[dict, int]:
    population = gen_snapshot["population"]
    champion = max(population, key=lambda p: p.get("fitness", -1.0))
    return champion, population.index(champion)


def run_fold(exp_dir: Path, target_gen: int) -> Optional[FoldResult]:
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        print(f"[RQ0] Skipping {exp_dir}: missing config.yaml")
        return None
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _mock_exp_manager(config)

    gen_file = _find_gen_file(exp_dir, target_gen)
    if gen_file is None:
        print(f"[RQ0] Skipping {exp_dir}: no generation snapshot found")
        return None
    with gen_file.open("r", encoding="utf-8") as f:
        snap = json.load(f)
    try:
        resolved_gen = int(gen_file.stem.split("_")[1])
    except Exception:
        resolved_gen = target_gen

    champion, champ_idx = _select_champion(snap)
    print(
        f"[RQ0] {exp_dir.name} gen={resolved_gen} champion={champ_idx} "
        f"val_qwk={champion.get('fitness', float('nan')):.4f}"
    )

    essay_set = int(config["data"]["essay_set"])
    fold = _infer_fold(exp_dir.name, fallback=0)

    all_data = asap_loader.load_asap(
        essay_set=essay_set,
        tsv_path=config["data"]["asap_path"],
        repo_root=_REPO_ROOT,
    )
    train_val_set, test_set = asap_loader.split_for_fold(all_data, fold)
    id_to_doc = {d["essay_id"]: d for d in train_val_set}

    # Vector store setup mirrors eval_5fold.py:73-80
    vector_store = SimpleVectorStore(model_name=config["rag"]["model_name"])
    vector_store.documents = train_val_set
    if config["rag"].get("enabled", False):
        vector_store.add_documents(train_val_set)

    rubric = champion["full_instruction"]
    ex_ids = champion["static_exemplar_ids"]
    exemplars: List[Dict] = []
    for eid in ex_ids:
        if eid in id_to_doc:
            exemplars.append(id_to_doc[eid])
        else:
            exemplars.append(train_val_set[0])

    ind = PromptIndividual(rubric, exemplars, config=config)
    use_rerank = config["rag"].get("use_rerank_test", True)
    test_qwk = ind.evaluate(test_set, vector_store, enable_rerank=use_rerank)

    y_true = np.asarray([item["domain1_score"] for item in test_set], dtype=int)
    y_pred = np.asarray(ind.last_pred_scores, dtype=int)

    per_essay = pd.DataFrame(
        {
            "dataset": "asap",
            "prompt_id": essay_set,
            "fold": fold,
            "essay_id": [item["essay_id"] for item in test_set],
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_err": np.abs(y_pred - y_true),
            "essay_len_words": [len(item["essay_text"].split()) for item in test_set],
        }
    )

    # Sanity check: recomputed QWK should match what ind.evaluate reported.
    recompute_qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    if abs(recompute_qwk - test_qwk) > 1e-6:
        print(
            f"[RQ0][warn] QWK mismatch for {exp_dir.name}: "
            f"ind.evaluate={test_qwk:.6f} recompute={recompute_qwk:.6f}"
        )

    return FoldResult(
        prompt=essay_set,
        fold=fold,
        exp_dir=str(exp_dir),
        gen=resolved_gen,
        champion_val_qwk=float(champion.get("fitness", float("nan"))),
        champion_test_qwk=float(test_qwk),
        per_essay=per_essay,
    )


# ---------------------------------------------------------------------------
# 3. Band analysis (cross-band / same-band)
# ---------------------------------------------------------------------------


def compute_band_edges(
    train_val_scores: np.ndarray, n_bands: int = 7
) -> np.ndarray:
    """Equal-frequency (quantile) band edges from training scores.

    Returns the ``n_bands - 1`` interior boundary values. Monotone increasing.
    Duplicate quantiles (due to tied integer scores) are collapsed.
    """
    if train_val_scores.size == 0:
        return np.array([])
    qs = np.linspace(0, 1, n_bands + 1)[1:-1]
    raw = np.quantile(train_val_scores, qs)
    # Drop duplicates while keeping order
    uniq, idx = np.unique(raw, return_index=True)
    return uniq[np.argsort(idx)]


def score_to_band(score: float, edges: np.ndarray) -> int:
    """Map a scalar score to a band index in [0, len(edges)]."""
    if edges.size == 0:
        return 0
    # np.searchsorted with side='right' gives class index under strict-left edges
    return int(np.searchsorted(edges, score, side="right"))


def enrich_per_essay(
    per_essay: pd.DataFrame, train_val_scores_by_prompt_fold: Dict[Tuple[int, int], np.ndarray]
) -> pd.DataFrame:
    """Augment per-essay rows with band indices and cross-band flag.

    ``cross_band`` is True iff the predicted score falls into a different
    band than the true score (``band_err >= 1``). This is the central
    Protocol-to-Score Misalignment signal.
    """
    out_frames: List[pd.DataFrame] = []
    for (prompt, fold), sub in per_essay.groupby(["prompt_id", "fold"]):
        scores = train_val_scores_by_prompt_fold.get((prompt, fold))
        if scores is None:
            scores = sub["y_true"].to_numpy()  # fallback — no leakage guarantees
        edges = compute_band_edges(scores, n_bands=7)
        sub = sub.copy()
        sub["band_true"] = sub["y_true"].map(lambda s: score_to_band(s, edges))
        sub["band_pred"] = sub["y_pred"].map(lambda s: score_to_band(s, edges))
        sub["band_err"] = (sub["band_pred"] - sub["band_true"]).abs()
        sub["cross_band"] = sub["band_err"] >= 1
        sub["band_edges"] = [edges.tolist()] * len(sub)
        out_frames.append(sub)
    return pd.concat(out_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Error decomposition tables
# ---------------------------------------------------------------------------


def decomp_table(per_essay: pd.DataFrame) -> pd.DataFrame:
    """Per-(prompt, fold) error decomposition.

    Core metric: ``cross_band_error_share`` = fraction of errors where
    ``band_pred != band_true``. This is the Protocol-to-Score Misalignment
    signal used for the RQ0 gate.
    """
    rows: List[Dict] = []
    for (prompt, fold), sub in per_essay.groupby(["prompt_id", "fold"]):
        n = len(sub)
        errors = sub[sub["abs_err"] > 0]
        n_err = len(errors)
        n_cross_err = int(errors["cross_band"].sum())
        n_sameband_err = n_err - n_cross_err
        rows.append(
            {
                "prompt_id": prompt,
                "fold": fold,
                "n_essays": n,
                "n_errors": n_err,
                "err_rate": n_err / max(n, 1),
                "n_cross_band_errors": n_cross_err,
                "n_same_band_errors": n_sameband_err,
                "cross_band_error_share": n_cross_err / max(n_err, 1),
                "off_by_1_band_share": float(
                    (errors["band_err"] == 1).sum() / max(n_err, 1)
                ),
                "off_by_2plus_band_share": float(
                    (errors["band_err"] >= 2).sum() / max(n_err, 1)
                ),
                "mean_band_distance_errors": float(errors["band_err"].mean())
                if n_err > 0 else 0.0,
                "mean_band_distance_all": float(sub["band_err"].mean()),
                "qwk": cohen_kappa_score(
                    sub["y_true"], sub["y_pred"], weights="quadratic"
                ) if n >= 2 and sub["y_true"].nunique() > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def overall_summary(decomp: pd.DataFrame) -> Dict:
    if decomp.empty:
        return {"status": "no_data"}
    total_err = int(decomp["n_errors"].sum())
    total_cross_err = int(decomp["n_cross_band_errors"].sum())
    total_same_err = int(decomp["n_same_band_errors"].sum())
    cross_band_error_share = total_cross_err / max(total_err, 1)
    return {
        "n_prompts": int(decomp["prompt_id"].nunique()),
        "n_folds": int(decomp["fold"].nunique()),
        "n_fold_runs": int(len(decomp)),
        "total_essays": int(decomp["n_essays"].sum()),
        "total_errors": total_err,
        "cross_band_error_share": cross_band_error_share,
        "same_band_error_share": total_same_err / max(total_err, 1),
        "off_by_1_band_share_mean": float(decomp["off_by_1_band_share"].mean()),
        "off_by_2plus_band_share_mean": float(decomp["off_by_2plus_band_share"].mean()),
        "mean_band_distance_errors_mean": float(
            decomp["mean_band_distance_errors"].mean()
        ),
        "mean_band_distance_all_mean": float(
            decomp["mean_band_distance_all"].mean()
        ),
        "mean_qwk": float(decomp["qwk"].mean(skipna=True)),
        "gate_threshold": 0.60,
        "gate_pass": bool(cross_band_error_share >= 0.60),
    }


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "RQ0 diagnostic: per-essay error decomposition (cross-band vs "
            "same-band) for existing WISE-AES fold runs."
        )
    )
    p.add_argument(
        "--gen", type=int, required=True,
        help="Target generation (e.g. 25). Falls back to the latest <=N if missing.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dirs", nargs="+",
        help="Explicit list of logs/exp_*_fold{N} directories (one per fold-run).",
    )
    g.add_argument(
        "--logs-root", type=str,
        help="Directory containing logs/exp_*_fold{N}; scan for latest per (prompt, fold).",
    )
    p.add_argument(
        "--prompts", type=int, nargs="*", default=None,
        help="Optional subset of prompt ids to include when using --logs-root.",
    )
    p.add_argument(
        "--out-dir", type=str, default="results/rq0",
        help="Output directory for per_essay_predictions.csv, error_decomp.csv, summary.json.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_cli()
    out_dir = _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Assemble exp dirs
    if args.dirs:
        exp_dirs = [Path(d) for d in args.dirs]
    else:
        root = Path(args.logs_root)
        if not root.is_absolute():
            root = _REPO_ROOT / root
        discovered = _discover_latest_per_fold(root, prompts=args.prompts)
        if not discovered:
            print(f"[RQ0] No exp_*_fold* dirs found under {root}")
            return 1
        exp_dirs = list(discovered.values())

    fold_results: List[FoldResult] = []
    for d in exp_dirs:
        d = d if d.is_absolute() else (_REPO_ROOT / d)
        if not d.exists():
            print(f"[RQ0] Missing dir: {d}")
            continue
        fr = run_fold(d, args.gen)
        if fr is not None:
            fold_results.append(fr)

    if not fold_results:
        print("[RQ0] No fold results produced.")
        return 1

    per_essay = pd.concat([fr.per_essay for fr in fold_results], ignore_index=True)

    # Build (prompt, fold) -> train_val_scores for leakage-safe band edges.
    train_val_scores_map: Dict[Tuple[int, int], np.ndarray] = {}
    for fr in fold_results:
        all_data = asap_loader.load_asap(
            essay_set=fr.prompt,
            tsv_path="data/raw/training_set_rel3.tsv",
            repo_root=_REPO_ROOT,
        )
        train_val, _ = asap_loader.split_for_fold(all_data, fr.fold)
        train_val_scores_map[(fr.prompt, fr.fold)] = np.asarray(
            [r["domain1_score"] for r in train_val], dtype=float
        )

    per_essay = enrich_per_essay(per_essay, train_val_scores_map)

    per_essay_path = out_dir / "per_essay_predictions.csv"
    per_essay.drop(columns=["band_edges"]).to_csv(per_essay_path, index=False)

    decomp = decomp_table(per_essay)
    decomp_path = out_dir / "error_decomp.csv"
    decomp.to_csv(decomp_path, index=False)

    summary = overall_summary(decomp)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 48)
    print("RQ0 Diagnostic Summary")
    print("=" * 48)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nOutputs:\n  {per_essay_path}\n  {decomp_path}\n  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
