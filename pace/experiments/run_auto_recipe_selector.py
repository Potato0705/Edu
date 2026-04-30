"""Run Prompt-Aware Automatic Recipe Selector (PARS).

The selector consumes existing PACE Layer-2 evidence caches. It does not call
WISE Layer-1 or rebuild Local-WISE predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pace.selector.auto_select import (  # noqa: E402
    SelectorTrainConfig,
    aggregate_summary,
    run_prompt_fold,
    write_csv,
)
from pace.selector.rule_gate import RuleGateConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt-aware automatic recipe selector for PACE-AES.")
    p.add_argument("--evidence-root", type=Path, default=None)
    p.add_argument("--evidence-search-roots", type=Path, nargs="*", default=[])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prompts", type=int, nargs="+", required=True)
    p.add_argument("--folds", type=int, nargs="+", required=True)
    p.add_argument("--mode", choices=["pars", "manual", "global"], default="pars")
    p.add_argument("--global-recipe", type=str, default="R1")
    p.add_argument("--recipe-library", type=str, default="v1")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mmd-warmup-epochs", type=int, default=3)
    p.add_argument("--mmd-margin", type=float, default=1.0)
    p.add_argument("--mmd-min-samples-per-band", type=int, default=4)
    p.add_argument("--mmd-sample-mode", choices=["boundary_only", "all"], default="boundary_only")
    p.add_argument("--mmd-sigma-mode", choices=["median_heuristic", "fixed"], default="median_heuristic")
    p.add_argument("--mmd-sigma", type=float, default=1.0)
    p.add_argument("--mmd-boundary-mode", choices=["uncertainty", "raw_boundary"], default="raw_boundary")
    p.add_argument("--mmd-uncertainty-threshold", type=float, default=0.15)
    p.add_argument("--mmd-project-dim", type=int, default=128)
    p.add_argument("--mmd-project-dropout", type=float, default=0.05)
    p.add_argument("--tie-qwk-tolerance", type=float, default=0.005)
    p.add_argument("--score-span-fine", type=float, default=5.0)
    p.add_argument("--score-span-large", type=float, default=20.0)
    p.add_argument("--off-by1-high", type=float, default=0.25)
    p.add_argument("--off-by2plus-high", type=float, default=0.12)
    p.add_argument("--mean-band-distance-high", type=float, default=0.40)
    p.add_argument("--adjacent-overlap-high", type=float, default=1.0)
    p.add_argument("--min-adjacent-pair-count", type=int, default=4)
    p.add_argument("--raw-mode-share-high", type=float, default=0.60)
    p.add_argument("--raw-std-frac-low", type=float, default=0.18)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--skip-missing", action="store_true")
    return p.parse_args()


def candidate_roots(args: argparse.Namespace) -> List[Path]:
    roots: List[Path] = []
    if args.evidence_root is not None:
        roots.append(args.evidence_root)
    roots.extend(args.evidence_search_roots)
    if not roots:
        roots.append(PROJECT_ROOT / "results")
    return roots


def find_evidence_path(roots: Sequence[Path], prompt: int, fold: int) -> Optional[Path]:
    def priority(path: Path) -> tuple[int, str]:
        text = str(path)
        if "pace_fixed_recipe_5fold" in text:
            rank = 0
        elif "pace_struct_runs" in text:
            rank = 1
        elif "pace_calibration_bf16" in text:
            rank = 2
        elif "pace_sweeps" in text:
            rank = 3
        else:
            rank = 4
        return rank, text

    direct_rel = Path(f"p{prompt}_fold{fold}") / "evidence_cache.pt"
    for root in roots:
        direct = root / direct_rel
        if direct.exists():
            return direct
    pattern = f"**/p{prompt}_fold{fold}/evidence_cache.pt"
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(root.glob(pattern))
        if matches:
            return sorted(matches, key=priority)[0]
    return None


def grouped_mean_rows(rows: Sequence[Dict[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    metric_cols = [
        "local_wise_qwk",
        "local_wise_mae",
        "local_wise_acc",
        "auto_pace_qwk",
        "auto_pace_mae",
        "auto_pace_acc",
        "delta_qwk",
        "delta_mae",
        "delta_acc",
    ]
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    out: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        payload: Dict[str, Any] = {group_key: key, "n_folds": len(group)}
        for col in metric_cols:
            vals = [float(r[col]) for r in group if col in r and math.isfinite(float(r[col]))]
            payload[col] = float(np.mean(vals)) if vals else float("nan")
        out.append(payload)
    return out


def overall_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = grouped_mean_rows(rows, "mode")
    if grouped:
        row = grouped[0]
        row["scope"] = "overall"
        return row
    return {"scope": "overall", "n_folds": 0}


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = candidate_roots(args)
    train_config = SelectorTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        seed=args.seed,
        mmd_warmup_epochs=args.mmd_warmup_epochs,
        mmd_margin=args.mmd_margin,
        mmd_min_samples_per_band=args.mmd_min_samples_per_band,
        mmd_sample_mode=args.mmd_sample_mode,
        mmd_sigma_mode=args.mmd_sigma_mode,
        mmd_sigma=args.mmd_sigma,
        mmd_boundary_mode=args.mmd_boundary_mode,
        mmd_uncertainty_threshold=args.mmd_uncertainty_threshold,
        mmd_project_dim=args.mmd_project_dim,
        mmd_project_dropout=args.mmd_project_dropout,
        tie_qwk_tolerance=args.tie_qwk_tolerance,
        device=args.device,
    )
    rule_config = RuleGateConfig(
        score_span_fine=args.score_span_fine,
        score_span_large=args.score_span_large,
        off_by1_high=args.off_by1_high,
        off_by2plus_high=args.off_by2plus_high,
        mean_band_distance_high=args.mean_band_distance_high,
        adjacent_overlap_high=args.adjacent_overlap_high,
        min_adjacent_pair_count=args.min_adjacent_pair_count,
        raw_mode_share_high=args.raw_mode_share_high,
        raw_std_frac_low=args.raw_std_frac_low,
    )

    feature_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    main_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for prompt in args.prompts:
        for fold in args.folds:
            evidence_path = find_evidence_path(roots, prompt, fold)
            if evidence_path is None:
                msg = f"missing evidence for P{prompt} fold{fold}"
                if args.skip_missing:
                    print(f"[pars] SKIP {msg}")
                    error_rows.append({"prompt": prompt, "fold": fold, "error": msg})
                    continue
                raise FileNotFoundError(msg)
            print(f"[pars] P{prompt} fold{fold} mode={args.mode} evidence={evidence_path}")
            features, decision, main_result = run_prompt_fold(
                evidence_path,
                args.out_dir,
                mode=args.mode,
                recipe_library=args.recipe_library,
                global_recipe=args.global_recipe,
                train_config=train_config,
                rule_config=rule_config,
                limit_train=args.limit_train,
                limit_val=args.limit_val,
                limit_test=args.limit_test,
            )
            feature_rows.append(features)
            decision_rows.append(decision)
            main_rows.append(main_result)
            print(
                "[pars] selected={recipe} inner_qwk={inner:.4f} "
                "test_qwk={test:.4f} delta={delta:.4f}".format(
                    recipe=decision["selected_recipe"],
                    inner=float(decision["inner_val_qwk"]),
                    test=float(main_result["auto_pace_qwk"]),
                    delta=float(main_result["delta_qwk"]),
                )
            )

    write_csv(args.out_dir / "selector_features.csv", feature_rows)
    write_csv(args.out_dir / "selector_decisions.csv", decision_rows)
    write_csv(args.out_dir / "five_fold_main_results.csv", main_rows)
    write_csv(args.out_dir / "five_fold_main_results_by_prompt.csv", grouped_mean_rows(main_rows, "prompt"))
    write_csv(args.out_dir / "five_fold_main_results_overall.csv", [overall_row(main_rows)])
    if error_rows:
        write_csv(args.out_dir / "selector_errors.csv", error_rows)

    summary = aggregate_summary(feature_rows, decision_rows, main_rows)
    summary.update(
        {
            "mode": args.mode,
            "recipe_library": args.recipe_library,
            "global_recipe": args.global_recipe if args.mode == "global" else None,
            "prompts": args.prompts,
            "folds": args.folds,
            "evidence_roots": [str(r) for r in roots],
            "train_config": train_config.__dict__,
            "rule_config": rule_config.__dict__,
            "n_errors": len(error_rows),
        }
    )
    with (args.out_dir / "selector_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[pars] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
