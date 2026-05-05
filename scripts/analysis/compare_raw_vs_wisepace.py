#!/usr/bin/env python
"""Compare a raw-only run with a WISE-PACE run using saved logs only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict


def load_run(exp_dir: Path) -> Dict[str, Any]:
    final_result = json.loads((exp_dir / "final_result.json").read_text(encoding="utf-8"))
    test_results = final_result.get("test_results", {})
    primary = test_results.get("raw_primary_candidate") or test_results.get("primary_candidate")
    candidate = (test_results.get("candidate_results") or {}).get(primary, {})
    audit = candidate.get("high_score_audit") or test_results.get("raw_primary_high_score_audit") or {}
    return {
        "exp_dir": str(exp_dir),
        "primary_candidate": primary,
        "raw_test_qwk": candidate.get("raw_test_qwk", test_results.get("raw_primary_test_qwk")),
        "val_qwk": candidate.get("validation_score", test_results.get("raw_primary_val_qwk")),
        "high_recall": audit.get("high_score_recall"),
        "max_recall": audit.get("max_score_recall"),
        "total_duration_sec": test_results.get("total_duration_sec"),
        "total_tokens_all": test_results.get("total_tokens_all"),
        "total_tokens": test_results.get("total_tokens"),
        "total_pace_tokens": test_results.get("total_pace_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-only-exp", required=True)
    parser.add_argument("--wisepace-exp", required=True)
    parser.add_argument("--out", default="raw_vs_wisepace_comparison.csv")
    args = parser.parse_args()

    rows = [
        {"run_type": "raw_only", **load_run(Path(args.raw_only_exp))},
        {"run_type": "wisepace", **load_run(Path(args.wisepace_exp))},
    ]
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
