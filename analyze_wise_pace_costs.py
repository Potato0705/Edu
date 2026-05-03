"""Cost and generation-to-target summaries for WISE-PACE runs.

Usage:
  python analyze_wise_pace_costs.py logs/exp_YYYYMMDD_HHMMSS_fold0 --targets 0.35 0.45
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_num(value):
    if value is None:
        return ""
    return _num(value)


def _pace_token_totals(population: list[dict]) -> dict:
    totals = {
        "pace_prompt_tokens": 0,
        "pace_completion_tokens": 0,
        "pace_representation_tokens": 0,
    }
    for individual in population:
        stats = individual.get("pace_cost_stats") or {}
        direct_total = (
            int(_num(stats.get("local_prompt_tokens"), 0))
            + int(_num(stats.get("local_completion_tokens"), 0))
            + int(_num(stats.get("local_representation_tokens"), 0))
        )
        if direct_total:
            totals["pace_prompt_tokens"] += int(_num(stats.get("local_prompt_tokens"), 0))
            totals["pace_completion_tokens"] += int(_num(stats.get("local_completion_tokens"), 0))
            totals["pace_representation_tokens"] += int(_num(stats.get("local_representation_tokens"), 0))
        else:
            usage = stats.get("local_usage") or {}
            totals["pace_prompt_tokens"] += int(_num(usage.get("prompt_tokens"), 0))
            totals["pace_completion_tokens"] += int(_num(usage.get("completion_tokens"), 0))
            totals["pace_representation_tokens"] += int(_num(usage.get("representation_tokens"), 0))
    totals["pace_tokens_total"] = (
        totals["pace_prompt_tokens"]
        + totals["pace_completion_tokens"]
        + totals["pace_representation_tokens"]
    )
    return totals


def load_generation_rows(exp_dir: Path) -> list[dict]:
    rows = []
    cumulative_seconds = 0.0
    cumulative_forwards = 0
    cumulative_tokens = 0
    cumulative_tokens_all = 0
    for path in sorted((exp_dir / "generations").glob("gen_*.json")):
        snap = _read_json(path)
        metrics = snap.get("metrics", {})
        population = snap.get("population", [])
        pace_tokens = _pace_token_totals(population)
        metrics_pace_tokens = {
            "pace_prompt_tokens": int(_num(metrics.get("pace_prompt_tokens"), 0)),
            "pace_completion_tokens": int(_num(metrics.get("pace_completion_tokens"), 0)),
            "pace_representation_tokens": int(_num(metrics.get("pace_representation_tokens"), 0)),
            "pace_tokens_total": int(_num(metrics.get("pace_tokens_total"), 0)),
        }
        if metrics_pace_tokens["pace_tokens_total"] >= pace_tokens["pace_tokens_total"]:
            pace_tokens = metrics_pace_tokens
        elif not pace_tokens["pace_tokens_total"]:
            pace_tokens = {
                "pace_prompt_tokens": metrics_pace_tokens["pace_prompt_tokens"],
                "pace_completion_tokens": metrics_pace_tokens["pace_completion_tokens"],
                "pace_representation_tokens": metrics_pace_tokens["pace_representation_tokens"],
                "pace_tokens_total": metrics_pace_tokens["pace_tokens_total"],
            }
        tokens_total = int(_num(metrics.get("tokens_total"), 0))
        tokens_total_all = int(_num(metrics.get("tokens_total_all"), tokens_total + pace_tokens["pace_tokens_total"]))
        cumulative_seconds += _num(metrics.get("duration_sec"))
        cumulative_forwards += int(_num(metrics.get("pace_total_forward_passes"), 0))
        cumulative_tokens += tokens_total
        cumulative_tokens_all += tokens_total_all
        best = max(population, key=lambda x: _num(x.get("fitness")), default={})
        rows.append(
            {
                "generation": snap.get("generation"),
                "best_fitness": _num(snap.get("best_qwk")),
                "best_raw_fitness": _num(best.get("raw_fitness")),
                "best_pace_evaluated": best.get("pace_fitness") is not None,
                "best_pace_fitness": _optional_num(best.get("pace_fitness")),
                "best_pace_raw_val_fitness": _optional_num(best.get("pace_raw_val_fitness")),
                "best_pace_raw_subset_fitness": _optional_num(best.get("pace_raw_fitness")),
                "best_pace_combined_subset_raw": _optional_num(best.get("pace_combined_subset_raw")),
                "best_combined_fitness": _num(best.get("combined_fitness")),
                "duration_sec": _num(metrics.get("duration_sec")),
                "tokens_total": tokens_total,
                "tokens_prompt": int(_num(metrics.get("tokens_prompt"), 0)),
                "tokens_completion": int(_num(metrics.get("tokens_completion"), 0)),
                "pace_prompt_tokens": pace_tokens["pace_prompt_tokens"],
                "pace_completion_tokens": pace_tokens["pace_completion_tokens"],
                "pace_representation_tokens": pace_tokens["pace_representation_tokens"],
                "pace_tokens_total": pace_tokens["pace_tokens_total"],
                "tokens_total_all": tokens_total_all,
                "pace_evaluated_count": int(_num(metrics.get("pace_evaluated_count"), 0)),
                "pace_early_rejected_count": int(_num(metrics.get("pace_early_rejected_count"), 0)),
                "pace_total_forward_passes": int(_num(metrics.get("pace_total_forward_passes"), 0)),
                "pace_total_inference_sec": _num(metrics.get("pace_total_inference_sec")),
                "anchor_mutation_count": int(_num(metrics.get("anchor_mutation_count"), 0)),
                "anchor_hidden_rerank_count": int(_num(metrics.get("anchor_hidden_rerank_count"), 0)),
                "cumulative_seconds": round(cumulative_seconds, 2),
                "cumulative_tokens": cumulative_tokens,
                "cumulative_tokens_all": cumulative_tokens_all,
                "cumulative_pace_forward_passes": cumulative_forwards,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generation_to_targets(rows: list[dict], targets: list[float]) -> list[dict]:
    out = []
    for target in targets:
        hit = next((row for row in rows if row["best_fitness"] >= target), None)
        out.append(
            {
                "target_qwk": target,
                "generation": hit["generation"] if hit else "",
                "cumulative_seconds": hit["cumulative_seconds"] if hit else "",
                "cumulative_tokens": hit["cumulative_tokens"] if hit else "",
                "cumulative_tokens_all": hit["cumulative_tokens_all"] if hit else "",
                "cumulative_pace_forward_passes": hit["cumulative_pace_forward_passes"] if hit else "",
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--targets", nargs="*", type=float, default=[0.35, 0.45, 0.55])
    args = parser.parse_args()

    if not (args.exp_dir / "generations").exists():
        raise SystemExit(f"Missing generations directory: {args.exp_dir}")

    rows = load_generation_rows(args.exp_dir)
    cost_path = args.exp_dir / "wise_pace_cost_summary.csv"
    target_path = args.exp_dir / "wise_pace_generation_to_target.csv"
    write_csv(cost_path, rows)
    write_csv(target_path, generation_to_targets(rows, args.targets))
    print(f"Wrote {cost_path} ({len(rows)} rows)")
    print(f"Wrote {target_path} ({len(args.targets)} targets)")
    if rows:
        best = max(rows, key=lambda r: r["best_fitness"])
        print(
            f"best generation={best['generation']} fitness={best['best_fitness']:.4f} "
            f"cumulative_minutes={best['cumulative_seconds'] / 60:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
