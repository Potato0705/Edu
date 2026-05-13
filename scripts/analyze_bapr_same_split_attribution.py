from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.llm_backend import LocalLlamaBackend  # noqa: E402
from scripts.run_anchor_budget_experiment import (  # noqa: E402
    bapr_to_anchor_record,
    instruction_from_config,
    load_asap_split,
    score_boundary_metrics,
    score_items,
    token_len,
)


ATTR_COLUMNS = [
    "method",
    "anchor_source",
    "val_diag_qwk",
    "val_sel_qwk",
    "test_qwk",
    "test_mae",
    "test_high_recall",
    "test_max_recall",
    "test_SCI",
    "test_range_coverage",
    "test_score_TV",
    "anchor_ids",
    "anchor_scores",
    "selected_reason",
]


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ATTR_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in ATTR_COLUMNS})


def resolve_run_dir(path: Path) -> Path:
    if (path / "bapr_parent_anchor_bank.json").exists():
        return path
    candidates = sorted(path.glob("bapr_repair_k_anchor_k*/bapr_parent_anchor_bank.json"))
    if not candidates:
        raise FileNotFoundError(f"Could not find BAPR run files under {path}")
    return candidates[0].parent


def metric_row(
    *,
    method: str,
    anchor_source: str,
    val_diag_qwk: Optional[float],
    val_sel_qwk: Optional[float],
    test_metrics: Dict[str, Any],
    bank: Dict[str, Any],
    selected_reason: str,
) -> Dict[str, Any]:
    return {
        "method": method,
        "anchor_source": anchor_source,
        "val_diag_qwk": val_diag_qwk,
        "val_sel_qwk": val_sel_qwk,
        "test_qwk": test_metrics.get("qwk"),
        "test_mae": test_metrics.get("mae"),
        "test_high_recall": test_metrics.get("high_recall"),
        "test_max_recall": test_metrics.get("max_recall"),
        "test_SCI": test_metrics.get("score_compression_index"),
        "test_range_coverage": test_metrics.get("range_coverage"),
        "test_score_TV": test_metrics.get("score_tv"),
        "anchor_ids": json.dumps(bank.get("anchor_ids", []), ensure_ascii=False),
        "anchor_scores": json.dumps(bank.get("anchor_scores", []), ensure_ascii=False),
        "selected_reason": selected_reason,
    }


def load_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {run_dir}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def score_parent_on_test(
    *,
    run_dir: Path,
    config: Dict[str, Any],
    parent_bank: Dict[str, Any],
    output_dir: Path,
    fold: int,
) -> Dict[str, Any]:
    score_min = int(config["data"]["score_min"])
    score_max = int(config["data"]["score_max"])
    split_manifest = config.get("fixed_split_manifest", {}).get("path")
    train, val, test = load_asap_split(
        config,
        fold,
        split_manifest=Path(split_manifest) if split_manifest else None,
    )
    del val
    instruction = instruction_from_config(config)
    by_id = {int(row["essay_id"]): row for row in train}
    missing = [int(essay_id) for essay_id in parent_bank["anchor_ids"] if int(essay_id) not in by_id]
    if missing:
        raise ValueError(f"Parent anchor IDs are not present in the reconstructed train split: {missing[:10]}")
    records = [
        bapr_to_anchor_record(
            {
                "essay_id": int(essay_id),
                "gold_score": int(by_id[int(essay_id)]["domain1_score"]),
                "prompt_id": int(config["data"]["essay_set"]),
                "token_length": token_len(str(by_id[int(essay_id)]["essay_text"])),
                "source_split": "train",
                "selection_score": 0.0,
                "selection_reason": "BAPR-A0 post-selection attribution",
                "essay_text": str(by_id[int(essay_id)]["essay_text"]),
            }
        )
        for essay_id in parent_bank["anchor_ids"]
    ]
    if bool(config.get("fake_scoring", False)):
        backend = None
    else:
        backend = LocalLlamaBackend(
            config=config,
            model_path=config.get("pace", {}).get("model_path", config.get("model", {}).get("name")),
            dtype=config.get("pace", {}).get("dtype", "bfloat16"),
            load_in_4bit=bool(config.get("pace", {}).get("load_in_4bit", False)),
        )
    rows, usage = score_items(backend, test, instruction, records, score_min, score_max)
    pred_path = output_dir / "bapr_a0_post_selection_test_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        keys = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    metrics = score_boundary_metrics(
        [int(row["gold_score"]) for row in rows],
        [int(row["pred_score"]) for row in rows],
        score_min,
        score_max,
    )
    metrics["post_selection_analysis_only"] = True
    metrics["usage"] = usage
    metrics["prediction_path"] = str(pred_path)
    with open(output_dir / "bapr_a0_post_selection_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def existing_parent_test_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = output_dir / "bapr_a0_post_selection_test_metrics.json"
    if path.exists():
        return read_json(path)
    return None


def render_markdown(
    *,
    run_dir: Path,
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    parent_test_scored: bool,
    baseline_note: str,
) -> str:
    by_method = {row["method"]: row for row in rows}
    parent = by_method.get("BAPR-A0", {})
    final = by_method.get("BAPR-A*", {})

    def delta(metric: str) -> str:
        p = parent.get(metric)
        f = final.get(metric)
        if p in (None, "") or f in (None, ""):
            return "n/a"
        try:
            return f"{float(f) - float(p):.4f}"
        except Exception:
            return "n/a"

    lines = [
        "# BAPR Same-Split Attribution\n\n",
        "## Scope\n",
        f"- run_dir: `{run_dir}`\n",
        f"- output_dir: `{output_dir}`\n",
        "- This is post-selection analysis only. Test metrics are not used to modify A*, guard decisions, parser, or config.\n",
        f"- parent_test_scored_posthoc: {parent_test_scored}\n",
        f"- baseline note: {baseline_note}\n\n",
        "## BAPR-A0 vs BAPR-A*\n",
        "| metric | BAPR-A0 | BAPR-A* | delta A*-A0 |\n",
        "|---|---:|---:|---:|\n",
    ]
    for metric in [
        "val_diag_qwk",
        "val_sel_qwk",
        "test_qwk",
        "test_mae",
        "test_high_recall",
        "test_max_recall",
        "test_SCI",
        "test_range_coverage",
        "test_score_TV",
    ]:
        lines.append(
            f"| {metric} | {parent.get(metric, '')} | {final.get(metric, '')} | {delta(metric)} |\n"
        )
    lines.extend(
        [
            "\n## Anchor Banks\n",
            f"- BAPR-A0 anchors: `{parent.get('anchor_ids', '')}`\n",
            f"- BAPR-A* anchors: `{final.get('anchor_ids', '')}`\n",
            f"- BAPR-A* selected_reason: `{final.get('selected_reason', '')}`\n\n",
            "## Interpretation Guardrails\n",
            "- If A* improves V_sel but not test, this remains MECHANISM_CHAIN_PASS_ONLY.\n",
            "- If A* improves test without V_sel support, do not treat it as method evidence.\n",
            "- High-tail and max-score metrics should be inspected separately because QWK can improve while high recall remains weak.\n",
        ]
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bapr-run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--skip-parent-test-scoring", action="store_true")
    args = parser.parse_args()

    run_dir = resolve_run_dir(Path(args.bapr_run_dir))
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(run_dir)
    parent_bank = read_json(run_dir / "bapr_parent_anchor_bank.json")
    final_bank = read_json(run_dir / "bapr_final_anchor_bank.json")
    parent_metrics = read_json(run_dir / "bapr_parent_metrics.json")
    failure_profile = read_json(run_dir / "bapr_failure_profile.json")
    score_metrics = read_json(run_dir / "score_boundary_metrics.json")
    final_val = score_metrics.get("val", {})
    final_test = score_metrics.get("test", {})

    parent_test_metrics = existing_parent_test_metrics(output_dir)
    parent_test_scored = False
    if parent_test_metrics is None:
        if args.skip_parent_test_scoring:
            parent_test_metrics = {}
        else:
            parent_test_metrics = score_parent_on_test(
                run_dir=run_dir,
                config=config,
                parent_bank=parent_bank,
                output_dir=output_dir,
                fold=args.fold,
            )
            parent_test_scored = True

    baseline_note = (
        "baseline lookup was requested but exact split baseline matching is not implemented; no non-BAPR baselines were mixed"
        if args.include_baselines
        else "baseline unavailable because exact split baseline was not requested"
    )

    rows = [
        metric_row(
            method="BAPR-A0",
            anchor_source="parent",
            val_diag_qwk=failure_profile.get("qwk"),
            val_sel_qwk=parent_metrics.get("qwk"),
            test_metrics=parent_test_metrics,
            bank=parent_bank,
            selected_reason="parent_anchor_bank",
        ),
        metric_row(
            method="BAPR-A*",
            anchor_source="final",
            val_diag_qwk=None,
            val_sel_qwk=final_val.get("qwk"),
            test_metrics=final_test,
            bank=final_bank,
            selected_reason=str(final_bank.get("selected_reason", "")),
        ),
    ]

    csv_path = output_dir / "bapr_p1_fold0_same_split_attribution.csv"
    md_path = output_dir / "bapr_p1_fold0_same_split_attribution.md"
    write_csv(csv_path, rows)
    md_path.write_text(
        render_markdown(
            run_dir=run_dir,
            output_dir=output_dir,
            rows=rows,
            parent_test_scored=parent_test_scored,
            baseline_note=baseline_note,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
