"""Plot Phase 2/4 anchor geometry trajectories from generation snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_best_rows(exp_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((exp_dir / "generations").glob("gen_*.json")):
        snap = json.loads(path.read_text(encoding="utf-8"))
        best = max(snap.get("population", []), key=lambda x: float(x.get("fitness", 0.0)), default={})
        rows.append(
            {
                "generation": snap.get("generation"),
                "fitness": _num(best.get("fitness")),
                "anchor_geometry_score": _num(best.get("anchor_geometry_score")),
                "anchor_separation": _num(best.get("anchor_separation")),
                "anchor_ordinal_consistency": _num(best.get("anchor_ordinal_consistency")),
                "anchor_monotonicity": _num(best.get("anchor_monotonicity")),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    args = parser.parse_args()
    rows = load_best_rows(args.exp_dir)
    if not rows:
        raise SystemExit("No generation snapshots found.")

    x = [r["generation"] for r in rows]
    plt.figure(figsize=(9, 5))
    for key in (
        "fitness",
        "anchor_geometry_score",
        "anchor_separation",
        "anchor_ordinal_consistency",
        "anchor_monotonicity",
    ):
        y = [r[key] for r in rows]
        if any(v is not None for v in y):
            plt.plot(x, y, marker="o", label=key)
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title("WISE-PACE Anchor Geometry Trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = args.exp_dir / "wise_pace_geometry_trajectory.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
