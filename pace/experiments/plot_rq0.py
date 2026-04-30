"""Plot RQ0 Figure 2 — Protocol-to-Score Misalignment.

Consumes ``results/rq0/per_essay_predictions.csv`` and
``results/rq0/error_decomp.csv`` produced by ``run_rq0_diagnostic.py`` and
emits:

* ``results/rq0/rq0_protocol_score_misalignment.png``
* ``results/rq0/rq0_protocol_score_misalignment.pdf``

The figure has three panels:
  (a) Histogram of per-essay |y_pred - y_true| stratified by band membership.
  (b) Boundary-vs-interior error share per prompt (stacked bars).
  (c) Cumulative error by band-distance (off-by-1, off-by-2, ...).

Usage::

    python -m pace.experiments.plot_rq0 \\
        --in-dir results/rq0 \\
        --out-stem results/rq0/rq0_protocol_score_misalignment
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _load(in_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_essay = pd.read_csv(in_dir / "per_essay_predictions.csv")
    decomp = pd.read_csv(in_dir / "error_decomp.csv")
    return per_essay, decomp


def _panel_abs_err_hist(ax: plt.Axes, per_essay: pd.DataFrame) -> None:
    errors = per_essay[per_essay["abs_err"] > 0]
    cross = errors[errors["cross_band"]]["abs_err"]
    same = errors[~errors["cross_band"]]["abs_err"]
    bins = np.arange(1, max(int(errors["abs_err"].max()) + 2, 3))
    ax.hist(
        [cross, same],
        bins=bins,
        label=["Cross-band error", "Same-band error"],
        stacked=True,
        edgecolor="white",
    )
    ax.set_xlabel("|y_pred - y_true|")
    ax.set_ylabel("# essays with error")
    ax.set_title("(a) Error magnitude distribution")
    ax.legend(frameon=False)


def _panel_cross_band_share_per_prompt(ax: plt.Axes, decomp: pd.DataFrame) -> None:
    agg = decomp.groupby("prompt_id").agg(
        n_cross_band_errors=("n_cross_band_errors", "sum"),
        n_same_band_errors=("n_same_band_errors", "sum"),
    ).reset_index()
    totals = agg["n_cross_band_errors"] + agg["n_same_band_errors"]
    totals = totals.replace(0, np.nan)
    cross_share = agg["n_cross_band_errors"] / totals
    same_share = agg["n_same_band_errors"] / totals

    x = np.arange(len(agg))
    ax.bar(x, cross_share, label="Cross-band", color="#d95f02")
    ax.bar(
        x, same_share, bottom=cross_share,
        label="Same-band", color="#7570b3",
    )
    ax.axhline(0.6, linestyle="--", color="black", linewidth=1, label="60% gate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in agg["prompt_id"]])
    ax.set_ylabel("Share of total errors")
    ax.set_title("(b) Cross-band vs same-band error share")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper right")


def _panel_band_distance_cdf(ax: plt.Axes, per_essay: pd.DataFrame) -> None:
    errors = per_essay[per_essay["abs_err"] > 0].copy()
    if errors.empty:
        ax.text(0.5, 0.5, "No errors found", ha="center", va="center")
        ax.set_title("(c) Band-distance CDF")
        return
    max_bd = int(errors["band_err"].max())
    xs = np.arange(0, max_bd + 1)
    cdf = [(errors["band_err"] <= k).mean() for k in xs]
    ax.plot(xs, cdf, marker="o")
    ax.set_xticks(xs)
    ax.set_xlabel("Band distance (|b_pred - b_true|)")
    ax.set_ylabel("CDF over errors")
    ax.set_ylim(0, 1.02)
    ax.set_title("(c) Errors concentrated at band distance 1")
    for x, y in zip(xs, cdf):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, default="results/rq0")
    p.add_argument("--out-stem", type=str, default="results/rq0/rq0_protocol_score_misalignment")
    return p.parse_args()


def main() -> int:
    args = _parse_cli()
    in_dir = Path(args.in_dir)
    per_essay, decomp = _load(in_dir)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _panel_abs_err_hist(axes[0], per_essay)
    _panel_cross_band_share_per_prompt(axes[1], decomp)
    _panel_band_distance_cdf(axes[2], per_essay)
    fig.suptitle("RQ0: Protocol-to-Score Misalignment (WISE-AES)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_stem = Path(args.out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    print(f"[plot_rq0] Wrote {out_stem.with_suffix('.png')} and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
