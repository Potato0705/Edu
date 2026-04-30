"""Build the per-fold anchor hidden-state cache for PACE-AES.

For each (prompt, fold) pair we:

1. Load the Layer-1 champion (I*, E*) via ``load_layer1_champion``.
2. Resolve the champion's ``static_exemplar_ids`` against the fold's
   train_val split (same KFold as ``eval_5fold.py``).
3. Classify the resolved exemplars into equal-width score bands
   (default: five anchors).
4. Score each anchor via :class:`LocalLlamaBackend` with **no peer
   exemplars** (``static_exemplars=[]``, ``dynamic_ex="(None)"``), capture
   the final-gen-token last-layer hidden state, and persist the anchor set
   through :class:`AnchorHiddenCache`.

The cache signature includes ``model_path | dtype | pool`` so switching
model weights invalidates existing caches automatically.

Usage
-----
::

    export TRANSFORMERS_OFFLINE=1

    uv run python -m pace.experiments.build_anchor_cache \\
        --model-path /data-ai/.../Meta-Llama-3.1-8B-Instruct \\
        --logs-root logs \\
        --prompts 1 2 3 4 5 6 7 8 \\
        --folds 0 1 2 3 4 \\
        --cache-root cache/pace_anchor_cache \\
        --dataset asap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pace.datasets import asap as asap_loader  # noqa: E402
from pace.llm_backend import (  # noqa: E402
    AnchorHiddenCache,
    AnchorRecord,
    LocalLlamaBackend,
    load_layer1_champion,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path", type=str, required=True,
        help="Local path to Meta-Llama-3.1-8B-Instruct (safetensors dir).",
    )
    p.add_argument("--logs-root", type=str, default="logs")
    p.add_argument("--prompts", type=int, nargs="+", required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--cache-root", type=str, default="cache/pace_anchor_cache")
    p.add_argument("--dataset", type=str, default="asap")
    p.add_argument("--gen", type=int, default=25)
    p.add_argument(
        "--force", action="store_true",
        help="Rebuild even if cache file exists with matching signature.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Resolve anchors + check Layer-1 logs; skip loading the model.",
    )
    p.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load Llama with bitsandbytes NF4 (fits ~6GB; needed for 8GB GPUs).",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=768,
        help="Lower this on small GPUs; 768 avoids truncating long CoT+JSON outputs.",
    )
    p.add_argument(
        "--num-anchor-bands",
        type=int,
        default=5,
        help="Number of equal-width anchor bands to cache per prompt/fold.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Exp-dir discovery (mirrors sanity script)
# ---------------------------------------------------------------------------


def _find_exp_dir(logs_root: Path, prompt: int, fold: int) -> Path:
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
# Anchor stratification
# ---------------------------------------------------------------------------


def _band_labels(num_bands: int) -> List[str]:
    if num_bands < 3:
        raise ValueError("num_bands must be >= 3")
    named = {
        3: ["low", "mid", "high"],
        5: ["very_low", "low", "mid", "high", "very_high"],
    }
    if num_bands in named:
        return named[num_bands]
    return [f"band_{i:02d}" for i in range(num_bands)]


def _band_index(score: int, score_min: int, score_max: int, num_bands: int) -> int:
    span = max(1, score_max - score_min)
    frac = float(score - score_min) / float(span)
    frac = min(max(frac, 0.0), 1.0)
    return min(num_bands - 1, int(frac * num_bands))


def _classify_band(score: int, score_min: int, score_max: int, labels: List[str]) -> str:
    return labels[_band_index(score, score_min, score_max, len(labels))]


def _pick_one_per_band(
    resolved: List[Dict],
    train_val: List[Dict],
    score_min: int,
    score_max: int,
    num_bands: int,
) -> List[AnchorRecord]:
    """Pick exactly one anchor per band.

    Prefer champion static exemplars, then fall back to the closest essay to
    the target band centre within the fold's train_val pool.
    """
    labels = _band_labels(num_bands)
    by_band: Dict[str, List[Dict]] = {label: [] for label in labels}
    for e in resolved:
        band = _classify_band(int(e["domain1_score"]), score_min, score_max, labels)
        by_band[band].append(e)

    span = float(score_max - score_min)
    band_centre = {
        label: score_min + span * ((idx + 0.5) / float(num_bands))
        for idx, label in enumerate(labels)
    }

    used_ids: set = set()
    anchors: List[AnchorRecord] = []
    for band in labels:
        pool = [e for e in by_band[band] if e["essay_id"] not in used_ids]
        if pool:
            pick = pool[0]
        else:
            target = band_centre[band]
            in_band = [
                e for e in train_val
                if _classify_band(
                    int(e["domain1_score"]),
                    score_min,
                    score_max,
                    labels,
                ) == band
                and e["essay_id"] not in used_ids
            ]
            search_pool = in_band if in_band else [
                e for e in train_val if e["essay_id"] not in used_ids
            ]
            if not search_pool:
                raise RuntimeError(
                    f"Cannot pick {band} anchor: train_val exhausted."
                )
            pick = min(
                search_pool,
                key=lambda e: abs(int(e["domain1_score"]) - target),
            )
        used_ids.add(pick["essay_id"])
        anchors.append(
            AnchorRecord(
                essay_id=pick["essay_id"],
                domain1_score=int(pick["domain1_score"]),
                band=band,
                essay_text=pick["essay_text"],
            )
        )
    return anchors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_cli()
    logs_root = Path(args.logs_root)
    cache = AnchorHiddenCache(args.cache_root)

    # Phase A: resolve all anchor tasks (cheap, no model needed yet).
    tasks: List[Dict] = []
    for prompt in args.prompts:
        all_data = asap_loader.load_asap(prompt, repo_root=_REPO_ROOT)
        for fold in args.folds:
            try:
                exp_dir = _find_exp_dir(logs_root, prompt, fold)
            except FileNotFoundError as exc:
                print(f"[anchor] SKIP P{prompt} fold{fold}: {exc}")
                continue
            champ = load_layer1_champion(exp_dir, gen=args.gen)
            cfg = champ["config"]
            score_min = int(cfg["data"]["score_min"])
            score_max = int(cfg["data"]["score_max"])
            train_val, _ = asap_loader.split_for_fold(all_data, fold)
            id_map = {e["essay_id"]: e for e in train_val}
            resolved = [
                id_map[i] for i in champ["static_exemplar_ids"] if i in id_map
            ]
            if not resolved:
                print(
                    f"[anchor] SKIP P{prompt} fold{fold}: no exemplar ids "
                    "resolved against train_val (log/split mismatch?)."
                )
                continue
            anchors = _pick_one_per_band(
                resolved,
                train_val,
                score_min,
                score_max,
                args.num_anchor_bands,
            )
            tasks.append(
                {
                    "prompt": prompt,
                    "fold": fold,
                    "exp_dir": exp_dir,
                    "config": cfg,
                    "instruction": champ["instruction"],
                    "score_min": score_min,
                    "score_max": score_max,
                    "anchors": anchors,
                }
            )
            band_ids = ", ".join(
                f"{a.band}:id{a.essay_id}(y={a.domain1_score})" for a in anchors
            )
            print(
                f"[anchor] P{prompt} fold{fold} exp={exp_dir.name} "
                f"score=[{score_min},{score_max}] "
                f"num_anchors={len(anchors)} anchors=[{band_ids}]"
            )

    if not tasks:
        print("[anchor] No tasks resolved. Exiting.")
        return 1

    if args.dry_run:
        print(f"[anchor] --dry-run: resolved {len(tasks)} tasks; skip model load.")
        return 0

    # Phase B: load model once, build every cache file.
    backend = LocalLlamaBackend(
        tasks[0]["config"],
        model_path=args.model_path,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )
    sig = backend.signature()
    print(f"[anchor] backend signature = {sig}")

    manifest_rows: List[Dict] = []
    for t in tasks:
        path = cache.path_for(args.dataset, t["prompt"], t["fold"])
        if path.exists() and not args.force:
            # AnchorHiddenCache.load_or_build handles signature validation
            pass
        entry = cache.load_or_build(
            dataset=args.dataset,
            prompt=t["prompt"],
            fold=t["fold"],
            anchors=t["anchors"],
            backend=backend,
            instruction=t["instruction"],
            score_min=t["score_min"],
            score_max=t["score_max"],
        )
        manifest_rows.append(
            {
                "dataset": args.dataset,
                "prompt_id": t["prompt"],
                "fold": t["fold"],
                "anchor_ids": entry.anchor_ids,
                "anchor_scores": entry.anchor_scores,
                "bands": entry.bands,
                "hidden_shape": list(entry.hidden.shape),
                "num_anchors": len(entry.anchor_ids),
                "model_path": entry.model_path,
                "backend_signature": entry.backend_signature,
                "cache_path": str(path),
            }
        )

    manifest_path = Path(args.cache_root) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)
    print(f"[anchor] Wrote manifest {manifest_path} ({len(manifest_rows)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
