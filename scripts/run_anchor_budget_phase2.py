from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_anchor_budget_results import (  # noqa: E402
    anchor_overlap_matrix,
    gap_to_full_static,
    load_run_artifacts,
    read_csv,
    same_k_comparison,
    write_csv,
)


PHASE2_CONFIGS = {
    1: "configs/anchor_budget_phase2_p1.yaml",
    2: "configs/anchor_budget_phase2_p2.yaml",
    7: "configs/anchor_budget_phase2_p7.yaml",
    8: "configs/anchor_budget_phase2_p8.yaml",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_command(cmd: Sequence[str], dry_run: bool = False) -> None:
    print("[Phase2] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), check=True)


def discover_latest_exp(root: Path, essay_set: int, fold: int) -> Path:
    candidates = sorted(
        root.glob(f"phase1_p{essay_set}_fold{fold}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No output directory for essay_set={essay_set}, fold={fold} under {root}")
    return candidates[0]


def enrich_prompt_rows(exp_dir: Path, essay_set: int) -> List[Dict[str, Any]]:
    rows = read_csv(exp_dir / "phase1_comparison_table.csv")
    enriched = []
    for row in rows:
        item = load_run_artifacts(exp_dir, row)
        item["prompt_id"] = essay_set
        item["essay_set"] = essay_set
        enriched.append(item)
    return enriched


def prompt_winner_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(int(row["prompt_id"]), []).append(row)
    out = []
    higher = ["val_qwk", "test_qwk", "high_recall", "max_recall", "SCI", "range_coverage"]
    lower = ["mae", "score_TV", "tokens"]
    for prompt_id, prompt_rows in sorted(by_prompt.items()):
        for metric in higher:
            best = max(prompt_rows, key=lambda r: float(r.get(metric, 0.0) or 0.0))
            out.append({"prompt_id": prompt_id, "metric": metric, "winner": best["method"], "k": best["k"], "value": best.get(metric)})
        for metric in lower:
            best = min(prompt_rows, key=lambda r: float(r.get(metric, 0.0) or 0.0))
            out.append({"prompt_id": prompt_id, "metric": metric, "winner": best["method"], "k": best["k"], "value": best.get(metric)})
    return out


def boundary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields = ["high_recall", "max_recall", "SCI", "range_coverage", "score_TV"]
    out = []
    for row in rows:
        out.append({k: row.get(k) for k in ["prompt_id", "method", "k", *fields]})
    return out


def phase2_same_prompt_deltas(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for prompt_id in sorted({int(r["prompt_id"]) for r in rows}):
        prompt_rows = [r for r in rows if int(r["prompt_id"]) == prompt_id]
        out.extend(same_k_comparison(prompt_rows))
        for row in out:
            row.setdefault("prompt_id", prompt_id)
    # same_k_comparison does not know prompt_id; recompute in a clearer way.
    fixed = []
    metrics = ["val_qwk", "test_qwk", "mae", "high_recall", "max_recall", "SCI", "range_coverage", "score_TV"]
    for prompt_id in sorted({int(r["prompt_id"]) for r in rows}):
        prompt_rows = [r for r in rows if int(r["prompt_id"]) == prompt_id]
        rep = next((r for r in prompt_rows if r["method"] == "representation_guided_k_anchor"), None)
        if not rep:
            continue
        for baseline_name in ["static_k_anchor", "retrieval_k_anchor"]:
            base = next((r for r in prompt_rows if r["method"] == baseline_name), None)
            if not base:
                continue
            item = {"prompt_id": prompt_id, "k": rep["k"], "baseline": baseline_name}
            for metric in metrics:
                rep_v = float(rep.get(metric, 0.0) or 0.0)
                base_v = float(base.get(metric, 0.0) or 0.0)
                item[f"delta_{metric}"] = rep_v - base_v
                item[f"rep_{metric}"] = rep_v
                item[f"baseline_{metric}"] = base_v
            item["representation_changed_anchor_choice"] = rep.get("representation_changed_anchor_choice")
            fixed.append(item)
    return fixed


def phase2_gap_to_full(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for prompt_id in sorted({int(r["prompt_id"]) for r in rows}):
        prompt_rows = [r for r in rows if int(r["prompt_id"]) == prompt_id]
        for row in gap_to_full_static(prompt_rows):
            row["prompt_id"] = prompt_id
            out.append(row)
    return out


def phase2_overlap(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for prompt_id in sorted({int(r["prompt_id"]) for r in rows}):
        prompt_rows = [r for r in rows if int(r["prompt_id"]) == prompt_id]
        for row in anchor_overlap_matrix(prompt_rows):
            row["prompt_id"] = prompt_id
            out.append(row)
    return out


def decide_phase2(rows: Sequence[Dict[str, Any]], deltas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rep_rows = [r for r in rows if r["method"] == "representation_guided_k_anchor"]
    changed_prompts = {
        int(r["prompt_id"])
        for r in rep_rows
        if str(r.get("representation_changed_anchor_choice")).lower() == "true"
    }
    metrics = ["val_qwk", "test_qwk", "high_recall", "SCI", "range_coverage"]
    lower_better = ["mae", "score_TV"]
    win_prompts = set()
    boundary_prompts = set()
    heavy_hurt = []
    for row in deltas:
        prompt = int(row["prompt_id"])
        if any(float(row.get(f"delta_{m}", 0.0) or 0.0) > 0 for m in metrics):
            win_prompts.add(prompt)
        if any(float(row.get(f"delta_{m}", 0.0) or 0.0) > 0 for m in ["high_recall", "SCI", "range_coverage"]):
            boundary_prompts.add(prompt)
        if float(row.get("delta_val_qwk", 0.0) or 0.0) < -0.10 and float(row.get("delta_test_qwk", 0.0) or 0.0) < -0.10:
            heavy_hurt.append(prompt)
        if any(float(row.get(f"delta_{m}", 0.0) or 0.0) < 0 for m in lower_better):
            win_prompts.add(prompt)
    if len(win_prompts) >= 2 and len(boundary_prompts) >= 2 and len(changed_prompts) >= 2 and not heavy_hurt:
        decision = "PASS"
        reason = "rep-guided k9 improves same-prompt baselines on multiple prompts and boundary metrics"
    elif len(win_prompts) >= 1 or len(boundary_prompts) >= 1:
        decision = "INCONCLUSIVE"
        reason = "rep-guided k9 has mixed cross-prompt signal"
    else:
        decision = "FAIL"
        reason = "rep-guided k9 does not improve same-prompt baselines"
    return {
        "decision": decision,
        "reason": reason,
        "win_prompts": sorted(win_prompts),
        "boundary_prompts": sorted(boundary_prompts),
        "changed_prompts": sorted(changed_prompts),
        "heavy_hurt_prompts": sorted(set(heavy_hurt)),
    }


def render_summary(
    output_root: Path,
    configs: Dict[int, Path],
    rows: Sequence[Dict[str, Any]],
    deltas: Sequence[Dict[str, Any]],
    decision: Dict[str, Any],
) -> str:
    lines = [
        "# Summary\n\n",
        "## Goal\n",
        "Phase 2 sanity checks whether representation-guided budgeted anchor selection is not P1-specific.\n\n",
        "## Setup\n",
        f"- commit: `{os.environ.get('WISE_PACE_COMMIT', '')}`\n",
        "- fold: 0\n",
        "- k: 9\n",
        "- methods: no_anchor, static_k_anchor, retrieval_k_anchor, representation_guided_k_anchor, full_static_anchor\n",
        "- final_pace_calibrated: false\n",
        "- test-time calibration: none\n",
        "- anchor pool: train split only\n\n",
        "## Configs\n",
    ]
    for prompt_id, path in configs.items():
        cfg = load_yaml(path)
        lines.append(
            f"- P{prompt_id}: `{path}` score_range={cfg['data']['score_min']}-{cfg['data']['score_max']}\n"
        )
    lines.extend([
        "\n## Results Table\n",
        "| prompt | method | k | val_qwk | test_qwk | mae | high_recall | max_recall | SCI | range_coverage | score_TV | tokens |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ])
    for row in rows:
        lines.append(
            f"| {row['prompt_id']} | {row['method']} | {row['k']} | {float(row.get('val_qwk', 0)):.4f} | "
            f"{float(row.get('test_qwk', 0)):.4f} | {float(row.get('mae', 0)):.4f} | "
            f"{float(row.get('high_recall', 0)):.4f} | {float(row.get('max_recall', 0)):.4f} | "
            f"{float(row.get('SCI', 0)):.4f} | {float(row.get('range_coverage', 0)):.4f} | "
            f"{float(row.get('score_TV', 0)):.4f} | {row.get('tokens')} |\n"
        )
    lines.extend([
        "\n## Decision\n",
        f"- **{decision['decision']}**: {decision['reason']}\n",
        f"- prompts with same-prompt wins: {decision['win_prompts']}\n",
        f"- prompts with boundary wins: {decision['boundary_prompts']}\n",
        f"- prompts where representation changed anchors: {decision['changed_prompts']}\n",
        f"- output_root: `{output_root}`\n",
    ])
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="*", type=int, default=[1, 2, 7, 8])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-root", default="logs/anchor_budget_phase2/cross_prompt_fold0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_configs = {p: Path(PHASE2_CONFIGS[p]) for p in args.prompts}
    if args.dry_run:
        plan = []
        for prompt_id, config_path in prompt_configs.items():
            cfg = load_yaml(config_path)
            plan.append(
                {
                    "prompt": prompt_id,
                    "config": str(config_path),
                    "score_range": [cfg["data"]["score_min"], cfg["data"]["score_max"]],
                    "command": [
                        sys.executable,
                        "scripts/run_anchor_budget_experiment.py",
                        "--config",
                        str(config_path),
                        "--fold",
                        str(args.fold),
                        "--methods",
                        "no_anchor",
                        "static_k_anchor",
                        "retrieval_k_anchor",
                        "representation_guided_k_anchor",
                        "full_static_anchor",
                        "--ks",
                        "9",
                    ],
                }
            )
        print(json.dumps({"output_root": str(output_root), "plan": plan}, ensure_ascii=False, indent=2))
        return

    all_rows: List[Dict[str, Any]] = []
    prompt_exp_dirs: Dict[int, str] = {}
    for prompt_id, config_path in prompt_configs.items():
        cfg = load_yaml(config_path)
        root = Path(cfg.get("output", {}).get("root", "logs/anchor_budget_phase2"))
        before = set(root.glob(f"phase1_p{prompt_id}_fold{args.fold}_*")) if root.exists() else set()
        run_command(
            [
                sys.executable,
                "scripts/run_anchor_budget_experiment.py",
                "--config",
                str(config_path),
                "--fold",
                str(args.fold),
                "--methods",
                "no_anchor",
                "static_k_anchor",
                "retrieval_k_anchor",
                "representation_guided_k_anchor",
                "full_static_anchor",
                "--ks",
                "9",
            ]
        )
        after = set(root.glob(f"phase1_p{prompt_id}_fold{args.fold}_*"))
        new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
        exp_dir = new_dirs[0] if new_dirs else discover_latest_exp(root, prompt_id, args.fold)
        prompt_exp_dirs[prompt_id] = str(exp_dir)
        all_rows.extend(enrich_prompt_rows(exp_dir, prompt_id))

    write_csv(output_root / "phase2_cross_prompt_comparison_table.csv", all_rows)
    winners = prompt_winner_rows(all_rows)
    deltas = phase2_same_prompt_deltas(all_rows)
    boundary = boundary_rows(all_rows)
    gaps = phase2_gap_to_full(all_rows)
    overlaps = phase2_overlap(all_rows)
    write_csv(output_root / "phase2_same_prompt_winners.csv", winners)
    write_csv(output_root / "phase2_same_prompt_rep_deltas.csv", deltas)
    write_csv(output_root / "phase2_boundary_metrics_by_prompt.csv", boundary)
    write_csv(output_root / "phase2_gap_to_full_static.csv", gaps)
    write_csv(output_root / "phase2_anchor_overlap_by_prompt.csv", overlaps)
    write_json(output_root / "phase2_exp_dirs.json", prompt_exp_dirs)
    decision = decide_phase2(all_rows, deltas)
    summary = render_summary(output_root, prompt_configs, all_rows, deltas, decision)
    (output_root / "new_mainline_phase2_cross_prompt_summary.md").write_text(summary, encoding="utf-8")
    Path("logs").mkdir(exist_ok=True)
    Path("logs/new_mainline_phase2_cross_prompt_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
