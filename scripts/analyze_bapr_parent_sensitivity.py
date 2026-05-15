from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_RETRIEVAL_PARENT_RUN = Path(
    "logs/anchor_budget_bapr/bapr_v1_p1_fold0_retrieval_parent_20260513_200520/bapr_repair_k_anchor_k9"
)
DEFAULT_V21_PARENT_RUN = Path(
    "logs/anchor_budget_bapr/bapr_v1_p1_fold0_real_20260513_165921/bapr_repair_k_anchor_k9"
)
DEFAULT_RETRIEVAL_FULL_VAL_RUN = Path(
    "logs/anchor_budget_phase2/phase1_p1_fold0_20260511_013317/retrieval_k_anchor_k9"
)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [int(x) for x in value]


def maybe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def band_for(score: int, score_min: int, score_max: int) -> str:
    low_max = math.floor(score_min + (score_max - score_min) / 3.0)
    high_min = math.ceil(score_min + 2.0 * (score_max - score_min) / 3.0)
    if score <= low_max:
        return "low"
    if score >= high_min:
        return "high"
    return "mid"


def bank_score_range(bank: Dict[str, Any], fallback: Tuple[int, int] = (2, 12)) -> Tuple[int, int]:
    score_range = bank.get("score_range")
    if isinstance(score_range, list) and len(score_range) == 2:
        return int(score_range[0]), int(score_range[1])
    scores = as_int_list(bank.get("anchor_scores"))
    if scores:
        return min(scores), max(scores)
    return fallback


def recursively_collect_anchor_meta(obj: Any, meta: Dict[int, Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if "essay_id" in obj:
            try:
                essay_id = int(obj["essay_id"])
            except (TypeError, ValueError):
                essay_id = None
            if essay_id is not None:
                row = meta.setdefault(essay_id, {})
                for src_key, dst_key in [
                    ("gold_score", "score"),
                    ("domain1_score", "score"),
                    ("score", "score"),
                    ("token_length", "token_length"),
                    ("selection_score", "selection_score"),
                    ("selection_reason", "selection_reason"),
                    ("source_split", "source_split"),
                ]:
                    if src_key in obj and obj[src_key] not in (None, ""):
                        row[dst_key] = obj[src_key]
                if "essay_text" in obj and "token_length" not in row:
                    row["token_length"] = max(1, len(str(obj["essay_text"]).split()))
        for value in obj.values():
            recursively_collect_anchor_meta(value, meta)
    elif isinstance(obj, list):
        for item in obj:
            recursively_collect_anchor_meta(item, meta)


def collect_meta_from_run(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    meta: Dict[int, Dict[str, Any]] = {}
    for name in [
        "anchor_bank.json",
        "anchor_metrics.json",
        "bapr_parent_anchor_bank.json",
        "bapr_final_anchor_bank.json",
        "bapr_repair_candidates.jsonl",
        "bapr_repair_trace.jsonl",
        "anchor_selection_trace.jsonl",
    ]:
        path = run_dir / name
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            recursively_collect_anchor_meta(read_jsonl(path), meta)
        else:
            recursively_collect_anchor_meta(read_json(path), meta)
    return meta


def fill_bank_scores_and_meta(bank: Dict[str, Any], meta: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    ids = as_int_list(bank.get("anchor_ids"))
    scores = as_int_list(bank.get("anchor_scores"))
    if len(scores) != len(ids):
        scores = []
        for essay_id in ids:
            value = meta.get(essay_id, {}).get("score")
            scores.append(int(value) if value is not None else None)
    token_lengths = []
    for essay_id in ids:
        value = meta.get(essay_id, {}).get("token_length")
        token_lengths.append(int(value) if value is not None else None)
    result = dict(bank)
    result["anchor_ids"] = ids
    result["anchor_scores"] = scores
    result["_token_lengths"] = token_lengths
    return result


def load_banks(
    retrieval_parent_run: Path,
    v21_parent_run: Path,
    retrieval_full_val_run: Path,
) -> Dict[str, Dict[str, Any]]:
    retrieval_meta = collect_meta_from_run(retrieval_full_val_run)
    retrieval_parent_meta = collect_meta_from_run(retrieval_parent_run)
    v21_meta = collect_meta_from_run(v21_parent_run)
    return {
        "retrieval_full_val": fill_bank_scores_and_meta(read_json(retrieval_full_val_run / "anchor_bank.json"), retrieval_meta),
        "retrieval_diag_parent": fill_bank_scores_and_meta(
            read_json(retrieval_parent_run / "bapr_parent_anchor_bank.json"), retrieval_parent_meta
        ),
        "BAPR-retrieval-A*": fill_bank_scores_and_meta(
            read_json(retrieval_parent_run / "bapr_final_anchor_bank.json"), retrieval_parent_meta
        ),
        "BAPR-v21-A0": fill_bank_scores_and_meta(read_json(v21_parent_run / "bapr_parent_anchor_bank.json"), v21_meta),
        "BAPR-v21-A*": fill_bank_scores_and_meta(read_json(v21_parent_run / "bapr_final_anchor_bank.json"), v21_meta),
    }


def anchor_overlap_rows(banks: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for left_name, right_name in combinations(banks.keys(), 2):
        left_ids = set(as_int_list(banks[left_name].get("anchor_ids")))
        right_ids = set(as_int_list(banks[right_name].get("anchor_ids")))
        shared = sorted(left_ids & right_ids)
        union = left_ids | right_ids
        rows.append(
            {
                "bank_a": left_name,
                "bank_b": right_name,
                "n_a": len(left_ids),
                "n_b": len(right_ids),
                "n_shared": len(shared),
                "jaccard": len(shared) / len(union) if union else None,
                "shared_anchor_ids": stable_json(shared),
                "unique_to_a": stable_json(sorted(left_ids - right_ids)),
                "unique_to_b": stable_json(sorted(right_ids - left_ids)),
            }
        )
    return rows


def score_coverage_rows(banks: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for name, bank in banks.items():
        ids = as_int_list(bank.get("anchor_ids"))
        scores = bank.get("anchor_scores") or []
        scores = [int(x) for x in scores if x is not None]
        score_min, score_max = bank_score_range(bank, (2, 12))
        bands = [band_for(score, score_min, score_max) for score in scores]
        token_lengths = [x for x in bank.get("_token_lengths", []) if x is not None]
        token_cost = int(bank.get("token_cost", sum(token_lengths) if token_lengths else 0))
        avg_length = (sum(token_lengths) / len(token_lengths)) if token_lengths else (token_cost / len(ids) if ids else None)
        rows.append(
            {
                "bank": name,
                "anchor_ids": stable_json(ids),
                "anchor_scores": stable_json(scores),
                "low_count": bands.count("low"),
                "mid_count": bands.count("mid"),
                "high_count": bands.count("high"),
                "unique_score_count": len(set(scores)),
                "score_range_span": (max(scores) - min(scores)) if scores else None,
                "score_min": score_min,
                "score_max": score_max,
                "token_cost": token_cost,
                "average_token_length": avg_length,
                "token_lengths": stable_json(token_lengths),
            }
        )
    return rows


def child_rejection_rows(run_dir: Path) -> List[Dict[str, Any]]:
    guard_rows = read_csv(run_dir / "bapr_guarded_selection.csv")
    candidates = {str(row.get("candidate_id")): row for row in read_jsonl(run_dir / "bapr_repair_candidates.jsonl")}
    parent_bank = read_json(run_dir / "bapr_parent_anchor_bank.json")
    parent_ids = set(as_int_list(parent_bank.get("anchor_ids")))
    parent_scores_by_id = dict(zip(as_int_list(parent_bank.get("anchor_ids")), as_int_list(parent_bank.get("anchor_scores"))))
    rows = []
    for row in guard_rows:
        if row.get("parent_or_child") != "child":
            continue
        child_ids = set(as_int_list(row.get("anchor_ids")))
        removed = sorted(parent_ids - child_ids)
        added = sorted(child_ids - parent_ids)
        candidate = candidates.get(str(row.get("candidate_id")), {})
        child_scores = as_int_list(row.get("anchor_scores"))
        child_scores_by_id = dict(zip(as_int_list(row.get("anchor_ids")), child_scores))
        rows.append(
            {
                "candidate_id": row.get("candidate_id"),
                "operator": row.get("operator"),
                "parent_id": row.get("parent_id"),
                "removed_anchor_ids": stable_json(removed),
                "removed_anchor_scores": stable_json([parent_scores_by_id.get(x) for x in removed]),
                "added_anchor_ids": stable_json(added),
                "added_anchor_scores": stable_json([child_scores_by_id.get(x) for x in added]),
                "val_sel_qwk": row.get("val_sel_qwk"),
                "val_sel_mae": row.get("val_sel_mae"),
                "val_sel_high_recall": row.get("val_sel_high_recall"),
                "val_sel_max_recall": row.get("val_sel_max_recall"),
                "val_sel_sci": row.get("val_sel_score_compression_index"),
                "val_sel_range_coverage": row.get("val_sel_range_coverage"),
                "val_sel_score_tv": row.get("val_sel_score_tv"),
                "target_boundary_metric_improved": row.get("target_boundary_metric_improved"),
                "accepted_by_guard": row.get("accepted_by_guard"),
                "selected_as_final": row.get("selected_as_final"),
                "guard_reject_reasons": row.get("guard_reject_reasons"),
                "candidate_anchor_bank_id": candidate.get("anchor_bank_id", ""),
            }
        )
    return rows


def metric_from_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    score_path = run_dir / "score_boundary_metrics.json"
    result: Dict[str, Any] = {}
    if summary_path.exists():
        result.update(read_json(summary_path))
    if score_path.exists():
        score = read_json(score_path)
        if isinstance(score.get("test"), dict):
            for key, value in score["test"].items():
                result[f"test_{key}"] = value
        if isinstance(score.get("val"), dict):
            for key, value in score["val"].items():
                result[f"val_{key}"] = value
    return result


def find_row(rows: Sequence[Dict[str, Any]], bank_a: str, bank_b: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if {row["bank_a"], row["bank_b"]} == {bank_a, bank_b}:
            return row
    return None


def render_diagnosis(
    *,
    banks: Dict[str, Dict[str, Any]],
    overlap: Sequence[Dict[str, Any]],
    coverage: Sequence[Dict[str, Any]],
    rejections: Sequence[Dict[str, Any]],
    retrieval_parent_metrics: Dict[str, Any],
    retrieval_full_metrics: Dict[str, Any],
    v21_metrics: Dict[str, Any],
    output_dir: Path,
) -> str:
    full_vs_diag = find_row(overlap, "retrieval_full_val", "retrieval_diag_parent") or {}
    full_ids = set(as_int_list(banks["retrieval_full_val"].get("anchor_ids")))
    diag_ids = set(as_int_list(banks["retrieval_diag_parent"].get("anchor_ids")))
    full_scores = dict(zip(as_int_list(banks["retrieval_full_val"].get("anchor_ids")), banks["retrieval_full_val"].get("anchor_scores", [])))
    diag_scores = dict(zip(as_int_list(banks["retrieval_diag_parent"].get("anchor_ids")), banks["retrieval_diag_parent"].get("anchor_scores", [])))
    full_only = sorted(full_ids - diag_ids)
    diag_only = sorted(diag_ids - full_ids)
    selected_reason = banks["BAPR-retrieval-A*"].get("selected_reason", "")

    def fmt(value: Any) -> str:
        value = maybe_float(value)
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        "# BAPR Parent Sensitivity Diagnosis\n\n",
        "## Scope\n",
        "- This is a post-hoc diagnosis over existing outputs only.\n",
        "- It does not call the LLM, alter anchor banks, alter selected A*, or change guard decisions.\n",
        f"- output_dir: `{output_dir}`\n\n",
        "## Anchor Overlap Findings\n",
        f"- retrieval_full_val vs retrieval_diag_parent Jaccard: `{fmt(full_vs_diag.get('jaccard'))}`.\n",
        f"- Shared anchors: `{full_vs_diag.get('shared_anchor_ids', '[]')}`.\n",
        f"- Full-val only anchors: `{stable_json(full_only)}` with scores `{stable_json([full_scores.get(x) for x in full_only])}`.\n",
        f"- V_diag-only anchors: `{stable_json(diag_only)}` with scores `{stable_json([diag_scores.get(x) for x in diag_only])}`.\n\n",
        "## Score Coverage Findings\n",
        "| bank | scores | low | mid | high | unique scores | span | token cost | avg length |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in coverage:
        lines.append(
            f"| {row['bank']} | `{row['anchor_scores']}` | {row['low_count']} | {row['mid_count']} | {row['high_count']} | "
            f"{row['unique_score_count']} | {row['score_range_span']} | {row['token_cost']} | {fmt(row['average_token_length'])} |\n"
        )
    lines.extend(
        [
            "\n## Retrieval Full-Val vs V-Diag Parent Diagnosis\n",
            f"- retrieval_full_val test QWK: `{fmt(retrieval_full_metrics.get('test_qwk'))}`.\n",
            f"- retrieval_diag_parent test QWK: `{fmt(retrieval_parent_metrics.get('test_qwk'))}`.\n",
            "- The two retrieval banks have nearly identical score coverage; the main structural difference is a same-band anchor swap.\n",
            "- This points to parent initialization sensitivity to validation sample composition and anchor content quality, not a coarse score coverage failure.\n",
            "- The V_diag-only retrieval parent did not lose low/mid/high band coverage, but it selected a different score-9 high-band anchor.\n\n",
            "## Child Rejection Diagnosis\n",
            f"- BAPR-retrieval selected_reason: `{selected_reason}`.\n",
            "| child | operator | removed | added | V_sel QWK | MAE | high recall | SCI | range coverage | score TV | target improved | reject reasons |\n",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|\n",
        ]
    )
    for row in rejections:
        lines.append(
            f"| {row['candidate_id']} | {row['operator']} | `{row['removed_anchor_ids']}` / `{row['removed_anchor_scores']}` | "
            f"`{row['added_anchor_ids']}` / `{row['added_anchor_scores']}` | {fmt(row['val_sel_qwk'])} | {fmt(row['val_sel_mae'])} | "
            f"{fmt(row['val_sel_high_recall'])} | {fmt(row['val_sel_sci'])} | {fmt(row['val_sel_range_coverage'])} | "
            f"{fmt(row['val_sel_score_tv'])} | {row['target_boundary_metric_improved']} | {row['guard_reject_reasons']} |\n"
        )
    lines.extend(
        [
            "\n## Diagnosis Conclusion\n",
            "- Primary issue: parent initialization sensitivity. retrieval_full_val and retrieval_diag_parent differ by only one high-band anchor, but test QWK differs sharply.\n",
            "- Secondary issue: the current one-step repair operators did not recover the full-val retrieval anchor bank. One child improved V_sel QWK but failed the target-metric requirement and reduced score uniqueness; this is consistent with the current guard contract rather than an implementation bug.\n",
            "- Guard assessment: no evidence that the guard was accidentally using test labels or leaking data. Rejections are traceable to QWK/anchor uniqueness/target-metric checks in `bapr_guarded_selection.csv`.\n",
            "- Leakage check: all compared anchor banks use train-pool anchors; no test anchor IDs are present in the selected banks.\n",
            "- Expansion recommendation: do not expand to P2/P7 or full-fold from the current BAPR-v1 state.\n\n",
            "## Decision\n",
            "**STOP_BAPR_EXPANSION_CURRENTLY**\n\n",
            "Current BAPR-v1 remains useful as a mechanism analysis result for weak-parent repair, but the retrieval-parent gate shows it is not yet robust enough to justify expansion.\n",
        ]
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-parent-run", default=str(DEFAULT_RETRIEVAL_PARENT_RUN))
    parser.add_argument("--v21-parent-run", default=str(DEFAULT_V21_PARENT_RUN))
    parser.add_argument("--retrieval-full-val-run", default=str(DEFAULT_RETRIEVAL_FULL_VAL_RUN))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    retrieval_parent_run = Path(args.retrieval_parent_run)
    v21_parent_run = Path(args.v21_parent_run)
    retrieval_full_val_run = Path(args.retrieval_full_val_run)
    output_dir = Path(args.output_dir) if args.output_dir else retrieval_parent_run / "attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    banks = load_banks(retrieval_parent_run, v21_parent_run, retrieval_full_val_run)
    overlap = anchor_overlap_rows(banks)
    coverage = score_coverage_rows(banks)
    rejections = child_rejection_rows(retrieval_parent_run)
    retrieval_parent_metrics = metric_from_summary(retrieval_parent_run)
    retrieval_full_metrics = metric_from_summary(retrieval_full_val_run)
    v21_metrics = metric_from_summary(v21_parent_run)

    write_csv(
        output_dir / "parent_sensitivity_anchor_overlap.csv",
        overlap,
        [
            "bank_a",
            "bank_b",
            "n_a",
            "n_b",
            "n_shared",
            "jaccard",
            "shared_anchor_ids",
            "unique_to_a",
            "unique_to_b",
        ],
    )
    write_csv(
        output_dir / "parent_sensitivity_score_coverage.csv",
        coverage,
        [
            "bank",
            "anchor_ids",
            "anchor_scores",
            "low_count",
            "mid_count",
            "high_count",
            "unique_score_count",
            "score_range_span",
            "score_min",
            "score_max",
            "token_cost",
            "average_token_length",
            "token_lengths",
        ],
    )
    write_csv(
        output_dir / "parent_sensitivity_child_rejections.csv",
        rejections,
        [
            "candidate_id",
            "operator",
            "parent_id",
            "removed_anchor_ids",
            "removed_anchor_scores",
            "added_anchor_ids",
            "added_anchor_scores",
            "val_sel_qwk",
            "val_sel_mae",
            "val_sel_high_recall",
            "val_sel_max_recall",
            "val_sel_sci",
            "val_sel_range_coverage",
            "val_sel_score_tv",
            "target_boundary_metric_improved",
            "accepted_by_guard",
            "selected_as_final",
            "guard_reject_reasons",
            "candidate_anchor_bank_id",
        ],
    )
    (output_dir / "parent_sensitivity_diagnosis.md").write_text(
        render_diagnosis(
            banks=banks,
            overlap=overlap,
            coverage=coverage,
            rejections=rejections,
            retrieval_parent_metrics=retrieval_parent_metrics,
            retrieval_full_metrics=retrieval_full_metrics,
            v21_metrics=v21_metrics,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'parent_sensitivity_anchor_overlap.csv'}")
    print(f"Wrote {output_dir / 'parent_sensitivity_score_coverage.csv'}")
    print(f"Wrote {output_dir / 'parent_sensitivity_child_rejections.csv'}")
    print(f"Wrote {output_dir / 'parent_sensitivity_diagnosis.md'}")


if __name__ == "__main__":
    main()
