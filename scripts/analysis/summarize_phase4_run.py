#!/usr/bin/env python
"""Summarize a WISE-PACE Phase-4 experiment directory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)

    final_result = load_json(exp_dir / "final_result.json")
    history = final_result.get("history") or load_json(exp_dir / "training_curve.json")
    test_results = final_result.get("test_results", {})
    high_rows = read_csv_rows(exp_dir / "high_score_audit.csv")
    parent_child = read_csv_rows(exp_dir / "parent_child_audit.csv")
    mutation_summary = read_csv_rows(exp_dir / "mutation_effect_summary.csv")

    best_generation = None
    if isinstance(history, list) and history:
        best_generation = max(history, key=lambda r: float(r.get("best_pareto") or r.get("best_raw_val") or -999))

    candidate_results = test_results.get("candidate_results", {})
    best_raw = candidate_results.get("best_raw_val") or {}
    best_guarded = candidate_results.get(test_results.get("raw_primary_candidate", "best_raw_guarded"), {})

    pace_summary = {
        "pace_tokens": test_results.get("total_pace_tokens"),
        "final_pace_tokens": test_results.get("final_pace_tokens"),
        "pace_evaluated_by_gen": [
            {"gen": row.get("gen"), "pace_evaluated_count": row.get("pace_evaluated_count")}
            for row in history
        ] if isinstance(history, list) else [],
    }
    cost_summary = {
        "total_duration_sec": test_results.get("total_duration_sec"),
        "total_tokens_all": test_results.get("total_tokens_all"),
        "total_tokens": test_results.get("total_tokens"),
        "total_prompt_tokens": test_results.get("total_prompt_tokens"),
        "total_completion_tokens": test_results.get("total_completion_tokens"),
    }

    summary = {
        "exp_dir": str(exp_dir),
        "best_generation": best_generation,
        "best_raw_candidate": {
            "raw_test_qwk": best_raw.get("raw_test_qwk"),
            "validation_score": best_raw.get("validation_score"),
            "high_score_audit": best_raw.get("high_score_audit"),
        },
        "best_guarded_candidate": {
            "raw_test_qwk": best_guarded.get("raw_test_qwk"),
            "validation_score": best_guarded.get("validation_score"),
            "high_score_audit": best_guarded.get("high_score_audit"),
        },
        "high_score_audit_rows": len(high_rows),
        "latest_high_score_audit": high_rows[-1] if high_rows else {},
        "mutation_effect_summary": mutation_summary,
        "parent_child_rows": len(parent_child),
        "pace_signal_summary": pace_summary,
        "cost_summary": cost_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
