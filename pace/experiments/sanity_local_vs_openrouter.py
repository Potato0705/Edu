"""Sanity: OpenRouter vs local Llama-3.1-8B y_raw consistency.

Phase-2 gate. For each (prompt, fold) pair we:

1. Load the Layer-1 champion (I*, E*) from the corresponding ``logs/exp_*``
   directory — same pathway as ``run_rq0_diagnostic.py``.
2. Reconstruct the **test** split for that fold (``KFold(5, shuffle=True,
   random_state=42)``) and deterministically sample ``--per-prompt`` essays.
3. Score each sampled essay under the **same** prompt template via
   :class:`OpenRouterBackend` and :class:`LocalLlamaBackend`.
4. Compare ``y_raw`` distributions (Pearson, Spearman, MAE, rank agreement,
   band agreement) and record per-essay rows for manual inspection.

Gate
----
* Pass if Pearson ≥ 0.85 **or** (Pearson ≥ 0.80 **and** Spearman ≥ 0.85).
* Soft warn on per-prompt mean drift > 0.5 (on the native score scale).

Usage
-----
::

    export TRANSFORMERS_OFFLINE=1              # server has no HF network
    export OPENROUTER_API_KEY=...

    uv run python -m pace.experiments.sanity_local_vs_openrouter \\
        --model-path /data-ai/.../Meta-Llama-3.1-8B-Instruct \\
        --logs-root logs \\
        --prompts 1 3 4 7 8 \\
        --per-prompt 10 \\
        --fold 0 \\
        --out-dir results/sanity
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pace.datasets import asap as asap_loader  # noqa: E402
from pace.llm_backend import (  # noqa: E402
    LocalLlamaBackend,
    OpenRouterBackend,
    ScoringRequest,
    load_layer1_champion,
)


# ---------------------------------------------------------------------------
# CLI / dir discovery
# ---------------------------------------------------------------------------


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path", type=str, required=True,
        help="Local path to Meta-Llama-3.1-8B-Instruct (safetensors dir).",
    )
    p.add_argument("--logs-root", type=str, default="logs")
    p.add_argument(
        "--prompts", type=int, nargs="+", required=True,
        help="ASAP prompt ids to sanity-check (e.g. 1 3 4 7 8).",
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--per-prompt", type=int, default=10)
    p.add_argument("--gen", type=int, default=25)
    p.add_argument(
        "--sample-pool", type=str, choices=["test", "train_val"], default="test",
        help="Which split to sample sanity essays from.",
    )
    p.add_argument("--out-dir", type=str, default="results/sanity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip LLM calls; only verify data access and dir layout.",
    )
    p.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load Llama with bitsandbytes NF4 (fits ~6GB; needed for 8GB GPUs).",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=768,
        help="Lower this on small GPUs; 768 avoids truncating long CoT+JSON outputs.",
    )
    return p.parse_args()


def _find_exp_dir(logs_root: Path, prompt: int, fold: int) -> Path:
    """Find the most recent logs/exp_*_fold{fold} that matches prompt.

    Matches wise_aes.py's config.yaml ``data.essay_set`` field.
    """
    candidates: List[Path] = []
    for d in sorted(logs_root.glob(f"exp_*_fold{fold}")):
        cfg_path = d / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            continue
        if int(cfg.get("data", {}).get("essay_set", -1)) == prompt:
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(
            f"No exp dir under {logs_root} for prompt={prompt} fold={fold}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_essays(
    all_data: List[Dict], fold: int, pool: str, n: int, seed: int
) -> List[Dict]:
    train_val, test = asap_loader.split_for_fold(all_data, fold)
    source = test if pool == "test" else train_val
    rng = random.Random(seed)
    if n >= len(source):
        return list(source)
    return rng.sample(source, n)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return _pearson(rx, ry)


def _summarize_pair(
    prompt: int, rows: List[Dict], score_min: int, score_max: int
) -> Dict:
    if not rows:
        return {
            "prompt_id": prompt, "n": 0, "pearson": float("nan"),
            "spearman": float("nan"), "mae": float("nan"),
            "mean_openrouter": float("nan"), "mean_local": float("nan"),
            "mean_drift": float("nan"),
        }
    op = np.array([r["y_openrouter"] for r in rows], dtype=float)
    lo = np.array([r["y_local"] for r in rows], dtype=float)
    pearson = _pearson(op, lo)
    spearman = _spearman(op, lo)
    mae = float(np.mean(np.abs(op - lo)))
    return {
        "prompt_id": prompt,
        "n": len(rows),
        "score_min": score_min,
        "score_max": score_max,
        "pearson": pearson,
        "spearman": spearman,
        "mae": mae,
        "mean_openrouter": float(np.mean(op)),
        "mean_local": float(np.mean(lo)),
        "mean_drift": float(np.mean(lo) - np.mean(op)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_cli()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_root = Path(args.logs_root)

    # First: resolve exp dirs + champions + data for every prompt (fail fast).
    plan: List[Dict] = []
    for prompt in args.prompts:
        exp_dir = _find_exp_dir(logs_root, prompt, args.fold)
        champ = load_layer1_champion(exp_dir, gen=args.gen)
        cfg = champ["config"]
        score_min = int(cfg["data"]["score_min"])
        score_max = int(cfg["data"]["score_max"])
        all_data = asap_loader.load_asap(prompt, repo_root=_REPO_ROOT)
        samples = _sample_essays(
            all_data, args.fold, args.sample_pool, args.per_prompt, args.seed + prompt
        )
        # static exemplars: resolve via essay_id against train_val
        train_val, _ = asap_loader.split_for_fold(all_data, args.fold)
        id_map = {e["essay_id"]: e for e in train_val}
        static_ex = [
            id_map[i] for i in champ["static_exemplar_ids"] if i in id_map
        ]
        plan.append(
            {
                "prompt": prompt,
                "exp_dir": exp_dir,
                "instruction": champ["instruction"],
                "static_exemplars": static_ex,
                "config": cfg,
                "score_min": score_min,
                "score_max": score_max,
                "samples": samples,
            }
        )
        print(
            f"[sanity] P{prompt} fold{args.fold} exp={exp_dir.name} "
            f"|static_ex|={len(static_ex)} |samples|={len(samples)} "
            f"range=[{score_min},{score_max}]"
        )

    if args.dry_run:
        print("[sanity] --dry-run: skipping LLM calls.")
        return 0

    # Initialise backends lazily (LocalLlama is heavy).
    # Use the config from the first prompt for the OpenRouter wrapper; call_llm
    # reads llm.* keys identically across prompts in this project.
    openrouter = OpenRouterBackend(plan[0]["config"])
    local = LocalLlamaBackend(
        plan[0]["config"],
        model_path=args.model_path,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )

    per_essay_rows: List[Dict] = []
    for task in plan:
        prompt = task["prompt"]
        for essay in task["samples"]:
            req = ScoringRequest(
                essay_id=essay["essay_id"],
                essay_text=essay["essay_text"],
                instruction=task["instruction"],
                static_exemplars=task["static_exemplars"],
                score_min=task["score_min"],
                score_max=task["score_max"],
                dynamic_ex="(None)",
            )
            res_or = openrouter.score(req)
            res_lo = local.score(req)
            per_essay_rows.append(
                {
                    "prompt_id": prompt,
                    "fold": args.fold,
                    "essay_id": essay["essay_id"],
                    "y_true": essay["domain1_score"],
                    "y_openrouter": res_or.y_raw,
                    "y_local": res_lo.y_raw,
                    "abs_diff": abs(res_or.y_raw - res_lo.y_raw),
                    "wall_openrouter": res_or.wallclock_sec,
                    "wall_local": res_lo.wallclock_sec,
                    "local_hidden_dim": (
                        int(res_lo.hidden.numel()) if res_lo.hidden is not None else 0
                    ),
                }
            )
            print(
                f"[sanity] P{prompt} essay={essay['essay_id']} "
                f"OR={res_or.y_raw} local={res_lo.y_raw} y_true={essay['domain1_score']}"
            )

    df = pd.DataFrame(per_essay_rows)
    per_path = out_dir / "per_essay.csv"
    df.to_csv(per_path, index=False)
    print(f"[sanity] Wrote {per_path}")

    # Per-prompt aggregates
    pp_rows: List[Dict] = []
    for task in plan:
        prompt = task["prompt"]
        prompt_rows = [r for r in per_essay_rows if r["prompt_id"] == prompt]
        pp_rows.append(
            _summarize_pair(prompt, prompt_rows, task["score_min"], task["score_max"])
        )
    pp_df = pd.DataFrame(pp_rows)
    pp_path = out_dir / "per_prompt_summary.csv"
    pp_df.to_csv(pp_path, index=False)
    print(f"[sanity] Wrote {pp_path}")

    # Overall summary + gate
    overall_rows = per_essay_rows
    op_all = np.array([r["y_openrouter"] for r in overall_rows], dtype=float)
    lo_all = np.array([r["y_local"] for r in overall_rows], dtype=float)
    pearson_overall = _pearson(op_all, lo_all)
    spearman_overall = _spearman(op_all, lo_all)
    mae_overall = float(np.mean(np.abs(op_all - lo_all))) if len(overall_rows) else float("nan")

    # Gate: (p>=0.85) or (p>=0.80 and s>=0.85)
    gate_pass = bool(
        (pearson_overall >= 0.85)
        or (pearson_overall >= 0.80 and spearman_overall >= 0.85)
    )

    # Drift warnings
    drift_warnings = [
        {"prompt_id": r["prompt_id"], "mean_drift": r["mean_drift"]}
        for r in pp_rows
        if isinstance(r["mean_drift"], float) and abs(r["mean_drift"]) > 0.5
    ]

    summary = {
        "n_total": len(overall_rows),
        "prompts": list(args.prompts),
        "fold": args.fold,
        "sample_pool": args.sample_pool,
        "per_prompt_n": args.per_prompt,
        "pearson_overall": pearson_overall,
        "spearman_overall": spearman_overall,
        "mae_overall": mae_overall,
        "gate_pass": gate_pass,
        "gate_rule": "pearson>=0.85 OR (pearson>=0.80 AND spearman>=0.85)",
        "drift_warnings": drift_warnings,
        "model_path": args.model_path,
        "load_in_4bit": args.load_in_4bit,
        "max_new_tokens": args.max_new_tokens,
        "backend_signature": local.signature(),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[sanity] Wrote {summary_path}")
    print(
        f"[sanity] pearson={pearson_overall:.4f} spearman={spearman_overall:.4f} "
        f"mae={mae_overall:.3f} gate_pass={gate_pass}"
    )
    if drift_warnings:
        print(f"[sanity] WARN drift on prompts: {drift_warnings}")
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
