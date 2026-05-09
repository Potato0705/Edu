"""Export reviewer-facing WISE-PACE artifacts from an experiment directory.

Produces a compact JSON and CSV with rubric text, anchor IDs/scores, trajectory
metrics, and representative diagnostic cases.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_individual(snapshot: dict) -> dict:
    return max(snapshot.get("population", []), key=lambda x: float(x.get("fitness", 0.0)))


def export(exp_dir: Path) -> tuple[Path, Path]:
    gen_paths = sorted((exp_dir / "generations").glob("gen_*.json"))
    if not gen_paths:
        raise FileNotFoundError(f"No gen_*.json files under {exp_dir / 'generations'}")

    rows = []
    artifacts = {
        "experiment": str(exp_dir),
        "generations": [],
    }
    for path in gen_paths:
        snap = _read_json(path)
        best = _best_individual(snap)
        item = {
            "generation": snap.get("generation"),
            "timestamp": snap.get("timestamp"),
            "best_fitness": best.get("fitness"),
            "raw_fitness": best.get("raw_fitness"),
            "pace_fitness": best.get("pace_fitness"),
            "combined_fitness": best.get("combined_fitness"),
            "anchor_geometry_score": best.get("anchor_geometry_score"),
            "dominant_error_type": best.get("dominant_error_type"),
            "static_exemplar_ids": best.get("static_exemplar_ids", []),
            "static_exemplar_scores": best.get("static_exemplar_scores", []),
            "static_exemplar_strata": best.get("static_exemplar_strata", []),
            "instruction": best.get("full_instruction", ""),
            "diagnostics": best.get("pace_diagnostics", []),
        }
        artifacts["generations"].append(item)
        rows.append(
            {
                "generation": item["generation"],
                "best_fitness": item["best_fitness"],
                "raw_fitness": item["raw_fitness"],
                "pace_fitness": item["pace_fitness"],
                "combined_fitness": item["combined_fitness"],
                "anchor_geometry_score": item["anchor_geometry_score"],
                "dominant_error_type": item["dominant_error_type"],
                "anchor_ids": ",".join(str(x) for x in item["static_exemplar_ids"]),
                "anchor_scores": ",".join(str(x) for x in item["static_exemplar_scores"]),
                "anchor_strata": ",".join(str(x) for x in item["static_exemplar_strata"]),
                "instruction_preview": item["instruction"][:240].replace("\n", " "),
            }
        )

    json_path = exp_dir / "wise_pace_artifacts.json"
    csv_path = exp_dir / "wise_pace_artifacts.csv"
    json_path.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    args = parser.parse_args()
    json_path, csv_path = export(args.exp_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
