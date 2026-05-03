"""Summarize WISE-PACE Phase 2 generation artifacts.

Usage:
  python analyze_phase2_trajectory.py logs/exp_YYYYMMDD_HHMMSS_fold0

Outputs:
  - phase2_trajectory.csv
  - phase2_anchor_mutations.csv
  - concise console summary
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _round_or_blank(value, ndigits=6):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return round(float(value), ndigits)
    return value


def load_generation_rows(exp_dir: Path) -> list[dict]:
    rows = []
    for gen_path in sorted((exp_dir / "generations").glob("gen_*.json")):
        snapshot = json.loads(gen_path.read_text(encoding="utf-8"))
        gen = snapshot["generation"]
        for ind in snapshot.get("population", []):
            row = {
                "generation": gen,
                "individual_id": ind.get("individual_id"),
                "fitness": _round_or_blank(ind.get("fitness")),
                "raw_fitness": _round_or_blank(ind.get("raw_fitness")),
                "pace_fitness": _round_or_blank(ind.get("pace_fitness")),
                "pace_raw_fitness": _round_or_blank(ind.get("pace_raw_fitness")),
                "pace_raw_val_fitness": _round_or_blank(ind.get("pace_raw_val_fitness")),
                "pace_combined_subset_raw": _round_or_blank(ind.get("pace_combined_subset_raw")),
                "combined_fitness": _round_or_blank(ind.get("combined_fitness")),
                "anchor_geometry_score": _round_or_blank(ind.get("anchor_geometry_score")),
                "anchor_separation": _round_or_blank(ind.get("anchor_separation")),
                "anchor_separation_raw": _round_or_blank(ind.get("anchor_separation_raw")),
                "anchor_ordinal_consistency": _round_or_blank(ind.get("anchor_ordinal_consistency")),
                "anchor_monotonicity": _round_or_blank(ind.get("anchor_monotonicity")),
                "dominant_error_type": ind.get("dominant_error_type") or "",
                "suggested_anchor_mutation_slot": ind.get("suggested_anchor_mutation_slot"),
                "pace_diagnostic_summary": json.dumps(
                    ind.get("pace_diagnostic_summary") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "early_rejection_metrics": json.dumps(
                    ind.get("early_rejection_metrics") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "cost_penalty": _round_or_blank(ind.get("cost_penalty")),
                "evidence_dim": ind.get("evidence_dim") or "",
                "static_exemplar_ids": ",".join(str(x) for x in ind.get("static_exemplar_ids", [])),
                "static_exemplar_scores": ",".join(str(x) for x in ind.get("static_exemplar_scores", [])),
                "static_exemplar_strata": ",".join(str(x) for x in ind.get("static_exemplar_strata", [])),
            }
            cost = ind.get("pace_cost_stats") or {}
            for key in (
                "anchor_inference_sec",
                "calib_inference_sec",
                "fitness_inference_sec",
                "calibrator_train_sec",
                "total_pace_sec",
                "n_calib",
                "n_fitness",
                "scoring_forward_passes",
                "representation_forward_passes",
                "total_local_forward_passes",
                "local_prompt_tokens",
                "local_completion_tokens",
                "local_representation_tokens",
            ):
                row[key] = _round_or_blank(cost.get(key))
            rows.append(row)
    return rows


def load_mutation_rows(exp_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((exp_dir / "generations").glob("anchor_mutations_gen_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for event in payload.get("events", []):
            rows.append(event)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python analyze_phase2_trajectory.py <exp_dir>", file=sys.stderr)
        return 2

    exp_dir = Path(sys.argv[1])
    if not (exp_dir / "generations").exists():
        print(f"Missing generations directory under {exp_dir}", file=sys.stderr)
        return 1

    generation_rows = load_generation_rows(exp_dir)
    mutation_rows = load_mutation_rows(exp_dir)

    trajectory_path = exp_dir / "phase2_trajectory.csv"
    mutation_path = exp_dir / "phase2_anchor_mutations.csv"
    write_csv(trajectory_path, generation_rows)
    write_csv(mutation_path, mutation_rows)

    print(f"Wrote {trajectory_path} ({len(generation_rows)} rows)")
    print(f"Wrote {mutation_path} ({len(mutation_rows)} rows)")

    by_gen = {}
    for row in generation_rows:
        gen = row["generation"]
        by_gen.setdefault(gen, []).append(row)
    for gen, rows in sorted(by_gen.items()):
        best = max(rows, key=lambda r: float(r["fitness"] or 0.0))
        print(
            f"gen={gen} best_ind={best['individual_id']} "
            f"fitness={best['fitness']} raw={best['raw_fitness']} "
            f"pace={best['pace_fitness']} anchor={best['anchor_geometry_score']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
