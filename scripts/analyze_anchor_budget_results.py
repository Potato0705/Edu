from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


METRICS = [
    "val_qwk",
    "test_qwk",
    "mae",
    "high_recall",
    "max_recall",
    "SCI",
    "range_coverage",
    "score_TV",
    "tokens",
    "runtime_sec",
]

CURVE_FILES = [
    "qwk_vs_k.csv",
    "high_recall_vs_k.csv",
    "sci_vs_k.csv",
    "range_coverage_vs_k.csv",
    "token_cost_vs_qwk.csv",
]


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def run_dir_for(exp_dir: Path, method: str, k: int) -> Path:
    if method in {"no_anchor", "full_static_anchor"}:
        return exp_dir / method
    return exp_dir / f"{method}_k{k}"


def load_run_artifacts(exp_dir: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    method = str(row.get("method", ""))
    k = to_int(row.get("k"))
    run_dir = run_dir_for(exp_dir, method, k)
    anchor_bank = read_json(run_dir / "anchor_bank.json")
    anchor_metrics = read_json(run_dir / "anchor_metrics.json")
    boundary_metrics = read_json(run_dir / "score_boundary_metrics.json")
    row = dict(row)
    row["run_dir"] = str(run_dir)
    row["anchor_ids"] = ",".join(str(x) for x in anchor_bank.get("anchor_ids", []))
    row["anchor_score_coverage"] = json.dumps(anchor_bank.get("score_coverage", {}), sort_keys=True)
    row["anchor_token_cost"] = anchor_bank.get("token_cost", "")
    row["anchor_count"] = anchor_metrics.get("anchor_count", anchor_bank.get("k", ""))
    row["avg_anchor_length"] = anchor_metrics.get("average_anchor_length", "")
    row["representation_changed_anchor_choice"] = anchor_bank.get(
        "representation_changed_anchor_choice",
        row.get("representation_changed_anchor_choice", False),
    )
    row["representation_features_used"] = json.dumps(anchor_bank.get("representation_features_used", []), ensure_ascii=False)
    row["test_prediction_distribution"] = json.dumps(
        (boundary_metrics.get("test") or {}).get("prediction_distribution", {}),
        sort_keys=True,
    )
    row["test_per_score_recall"] = json.dumps(
        (boundary_metrics.get("test") or {}).get("per_score_recall", {}),
        sort_keys=True,
    )
    return row


def same_k_comparison(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(r["method"], to_int(r["k"])): r for r in rows}
    out = []
    for k in sorted({to_int(r["k"]) for r in rows if r["method"].endswith("_k_anchor")}):
        rep = by_key.get(("representation_guided_k_anchor", k))
        if not rep:
            continue
        for baseline in ["static_k_anchor", "stratified_k_anchor", "retrieval_k_anchor"]:
            base = by_key.get((baseline, k))
            if not base:
                continue
            item = {"k": k, "baseline": baseline}
            for metric in METRICS:
                item[f"rep_{metric}"] = rep.get(metric, "")
                item[f"baseline_{metric}"] = base.get(metric, "")
                if metric in rep and metric in base:
                    item[f"delta_{metric}"] = to_float(rep.get(metric)) - to_float(base.get(metric))
            item["representation_changed_anchor_choice"] = rep.get("representation_changed_anchor_choice", "")
            item["rep_anchor_ids"] = rep.get("anchor_ids", "")
            item["baseline_anchor_ids"] = base.get("anchor_ids", "")
            out.append(item)
    return out


def gap_to_full_static(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    full = next((r for r in rows if r["method"] == "full_static_anchor"), None)
    if not full:
        return []
    out = []
    for row in rows:
        item = {"method": row["method"], "k": row["k"]}
        for metric in METRICS:
            item[metric] = row.get(metric, "")
            item[f"full_static_{metric}"] = full.get(metric, "")
            item[f"gap_to_full_static_{metric}"] = to_float(row.get(metric)) - to_float(full.get(metric))
        out.append(item)
    return out


def method_rankings(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranking_rows = []
    lower_is_better = {"mae", "score_TV", "tokens", "runtime_sec"}
    for metric in METRICS:
        ranked = sorted(
            rows,
            key=lambda r: to_float(r.get(metric)),
            reverse=metric not in lower_is_better,
        )
        for rank, row in enumerate(ranked, start=1):
            ranking_rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "method": row.get("method"),
                    "k": row.get("k"),
                    "value": row.get(metric),
                }
            )
    return ranking_rows


def anchor_overlap_matrix(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed = []
    for row in rows:
        ids = {x for x in str(row.get("anchor_ids", "")).split(",") if x}
        parsed.append((f"{row.get('method')}_k{row.get('k')}", ids))
    out = []
    for name_a, ids_a in parsed:
        for name_b, ids_b in parsed:
            union = ids_a | ids_b
            inter = ids_a & ids_b
            out.append(
                {
                    "run_a": name_a,
                    "run_b": name_b,
                    "n_overlap": len(inter),
                    "jaccard": len(inter) / len(union) if union else 1.0,
                    "overlap_ids": ",".join(sorted(inter, key=lambda x: int(x) if x.isdigit() else x)),
                }
            )
    return out


def load_curve_files(exp_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    return {name: read_csv(exp_dir / name) for name in CURVE_FILES}


def decision_report(
    exp_dir: Path,
    rows: Sequence[Dict[str, Any]],
    same_k: Sequence[Dict[str, Any]],
    gap_rows: Sequence[Dict[str, Any]],
    curve_rows: Dict[str, List[Dict[str, Any]]],
) -> str:
    rep_rows = [r for r in rows if r.get("method") == "representation_guided_k_anchor"]
    changed = [r for r in rep_rows if str(r.get("representation_changed_anchor_choice")).lower() == "true"]
    wins = []
    boundary_wins = []
    for row in same_k:
        baseline = row["baseline"]
        k = row["k"]
        if to_float(row.get("delta_val_qwk")) > 0:
            wins.append(f"k={k} val_qwk > {baseline}")
        if to_float(row.get("delta_high_recall")) > 0:
            boundary_wins.append(f"k={k} high_recall > {baseline}")
        if to_float(row.get("delta_SCI")) > 0:
            boundary_wins.append(f"k={k} SCI > {baseline}")
        if to_float(row.get("delta_range_coverage")) > 0:
            boundary_wins.append(f"k={k} range_coverage > {baseline}")
    full = next((r for r in rows if r.get("method") == "full_static_anchor"), None)
    close_to_full = []
    if full:
        full_qwk = to_float(full.get("test_qwk"))
        for rep in rep_rows:
            if full_qwk and to_float(rep.get("test_qwk")) >= full_qwk - 0.03:
                close_to_full.append(f"k={rep.get('k')} within 0.03 raw test QWK of full_static")

    if changed and (wins or boundary_wins or close_to_full):
        decision = "PASS"
        reason = "representation changed anchor choice and showed at least one same-budget or boundary/full-static signal"
    elif changed and not (wins or boundary_wins or close_to_full):
        decision = "FAIL"
        reason = "representation changed anchor choice but did not improve same-budget or boundary metrics"
    elif wins or boundary_wins:
        decision = "INCONCLUSIVE"
        reason = "some improvements appeared, but representation did not change anchor choice"
    else:
        decision = "FAIL"
        reason = "representation-guided anchors did not beat static/stratified/retrieval baselines"

    lines = [
        "# Anchor Budget Phase 1 Decision Report\n\n",
        "## Inputs\n",
        f"- exp_dir: `{exp_dir}`\n",
        "- final scoring: raw LLM score under selected anchor protocol\n",
        "- test-time calibration: none\n\n",
        "## Loaded Files\n",
    ]
    for name in ["phase1_comparison_table.csv", *CURVE_FILES]:
        exists = (exp_dir / name).exists()
        n_rows = len(curve_rows.get(name, [])) if name in curve_rows else len(rows)
        lines.append(f"- `{name}`: {'found' if exists else 'missing'} ({n_rows} rows)\n")
    lines.extend([
        "\n",
        "## Same-k Result\n",
        f"- validation wins over same-k baselines: {wins or 'none'}\n",
        f"- boundary/coverage wins over same-k baselines: {boundary_wins or 'none'}\n",
        f"- close to full static: {close_to_full or 'none'}\n",
        f"- representation changed anchor choice: {bool(changed)} ({len(changed)} runs)\n\n",
        "## Decision\n",
        f"- **{decision}**: {reason}\n\n",
        "## Required Checks\n",
        "- rep-guided beats static / stratified / retrieval under same k: see `same_k_comparison.csv`\n",
        "- rep-guided approaches full_static: see `gap_to_full_static.csv`\n",
        "- high recall / SCI / range coverage: see same-k deltas and rankings\n",
        "- anchor choice actually changed: `representation_changed_anchor_choice`\n",
        "- anchor overlap: see `anchor_overlap_matrix.csv`\n",
    ])
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)
    table_path = exp_dir / "phase1_comparison_table.csv"
    rows = read_csv(table_path)
    if not rows:
        raise FileNotFoundError(f"No rows found in {table_path}")
    enriched = [load_run_artifacts(exp_dir, row) for row in rows]
    curve_rows = load_curve_files(exp_dir)
    same_k = same_k_comparison(enriched)
    gap_rows = gap_to_full_static(enriched)
    rankings = method_rankings(enriched)
    overlap = anchor_overlap_matrix(enriched)
    write_csv(exp_dir / "same_k_comparison.csv", same_k)
    write_csv(exp_dir / "gap_to_full_static.csv", gap_rows)
    write_csv(exp_dir / "method_rankings_by_metric.csv", rankings)
    write_csv(exp_dir / "anchor_overlap_matrix.csv", overlap)
    report = decision_report(exp_dir, enriched, same_k, gap_rows, curve_rows)
    (exp_dir / "anchor_budget_phase1_decision_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
