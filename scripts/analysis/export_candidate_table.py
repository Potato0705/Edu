#!/usr/bin/env python
"""Export compact CSV tables from a Phase-4 experiment directory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", nargs="?")
    parser.add_argument("--exp-dir", dest="exp_dir_flag")
    args = parser.parse_args()
    exp_dir_arg = args.exp_dir_flag or args.exp_dir
    if not exp_dir_arg:
        parser.error("exp_dir is required, either positionally or via --exp-dir")
    exp_dir = Path(exp_dir_arg)
    final_result = json.loads((exp_dir / "final_result.json").read_text(encoding="utf-8"))
    test_results = final_result.get("test_results", {})
    candidate_results = test_results.get("candidate_results", {})

    candidate_rows = []
    final_rows = []
    for label, result in candidate_results.items():
        audit = result.get("high_score_audit") or {}
        row = {
            "label": label,
            "duplicate_of": result.get("duplicate_of"),
            "validation_score": result.get("validation_score"),
            "selection_score": result.get("selection_score"),
            "primary_candidate": test_results.get("primary_candidate"),
            "final_primary_policy": test_results.get("final_primary_policy"),
            "raw_test_qwk": result.get("raw_test_qwk"),
            "raw_test_mae": result.get("raw_test_mae"),
            "source_generation": result.get("source_generation"),
            "source_index": result.get("source_index"),
            "static_exemplar_ids": result.get("static_exemplar_ids"),
            "static_exemplar_scores": result.get("static_exemplar_scores"),
            "high_score_recall": audit.get("high_score_recall"),
            "max_score_recall": audit.get("max_score_recall"),
            "score_distribution_tv": audit.get("score_distribution_tv"),
            "validation_split": test_results.get("validation_split"),
        }
        candidate_rows.append(row)
        final_rows.append({**row, "selection_policy": test_results.get("selection_policy")})

    history = final_result.get("history") or []
    write_csv(exp_dir / "candidate_summary.csv", candidate_rows)
    write_csv(exp_dir / "generation_curve.csv", history)
    write_csv(exp_dir / "final_candidate_comparison.csv", final_rows)
    print(f"Wrote candidate_summary.csv, generation_curve.csv, final_candidate_comparison.csv in {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
