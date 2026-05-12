from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


REBALANCE_SCORE_BANDS = "REBALANCE_SCORE_BANDS"
REPLACE_WORST_BAND_ANCHOR = "REPLACE_WORST_BAND_ANCHOR"
DECOMPRESS_EXTREME_ANCHORS = "DECOMPRESS_EXTREME_ANCHORS"
REMOVE_CONFUSING_OR_REDUNDANT_ANCHOR = "REMOVE_CONFUSING_OR_REDUNDANT_ANCHOR"


OPERATOR_TARGET_METRICS = {
    REBALANCE_SCORE_BANDS: ["range_coverage", "score_tv", "worst_band_mae"],
    REPLACE_WORST_BAND_ANCHOR: ["worst_band_mae", "score_tv"],
    DECOMPRESS_EXTREME_ANCHORS: [
        "high_recall",
        "high_tail_under_score_rate",
        "max_recall",
        "max_score_under_score_rate",
        "range_coverage",
        "score_compression_index",
    ],
    REMOVE_CONFUSING_OR_REDUNDANT_ANCHOR: ["score_tv", "worst_band_mae", "score_compression_index"],
}


def stable_hash(payload: Any, n: int = 12) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def id_list_hash(ids: Sequence[int]) -> str:
    return hashlib.md5(",".join(map(str, ids)).encode("utf-8")).hexdigest()[:16]


def token_len(text: str) -> int:
    return max(1, len(str(text).split()))


def tokenize_simple(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{3,}", str(text).lower()))


def band_for(score: int, score_min: int, score_max: int) -> str:
    span = max(1, score_max - score_min)
    low_cut = score_min + span / 3.0
    high_cut = score_min + 2.0 * span / 3.0
    if score <= low_cut:
        return "low"
    if score >= high_cut:
        return "high"
    return "mid"


def score_distribution(values: Sequence[int], score_min: int, score_max: int) -> Dict[str, int]:
    counts = {str(s): 0 for s in range(score_min, score_max + 1)}
    for value in values:
        key = str(int(value))
        if key in counts:
            counts[key] += 1
    return counts


def tv_distance(true_counts: Dict[str, int], pred_counts: Dict[str, int]) -> float:
    n_true = max(1, sum(true_counts.values()))
    n_pred = max(1, sum(pred_counts.values()))
    keys = sorted(set(true_counts) | set(pred_counts), key=lambda x: int(x))
    return 0.5 * sum(abs(true_counts.get(k, 0) / n_true - pred_counts.get(k, 0) / n_pred) for k in keys)


def adaptive_high_threshold(y_true: Sequence[int], score_min: int, score_max: int) -> int:
    if not y_true:
        return score_max
    q75 = int(math.ceil(float(np.percentile([int(x) for x in y_true], 75))))
    top_band = int(math.ceil(score_min + 2.0 * max(1, score_max - score_min) / 3.0))
    return max(min(q75, score_max), top_band)


def score_metrics(y_true: Sequence[int], y_pred: Sequence[int], score_min: int, score_max: int) -> Dict[str, Any]:
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]
    true_counts = score_distribution(y_true, score_min, score_max)
    pred_counts = score_distribution(y_pred, score_min, score_max)
    high_threshold = adaptive_high_threshold(y_true, score_min, score_max)
    high_idx = [i for i, y in enumerate(y_true) if y >= high_threshold]
    max_idx = [i for i, y in enumerate(y_true) if y == score_max]
    high_recall = sum(1 for i in high_idx if y_pred[i] >= high_threshold) / len(high_idx) if high_idx else 1.0
    max_recall = sum(1 for i in max_idx if y_pred[i] == score_max) / len(max_idx) if max_idx else 1.0
    gold_std = float(np.std(y_true)) if y_true else 0.0
    pred_std = float(np.std(y_pred)) if y_pred else 0.0
    possible = max(1, score_max - score_min + 1)
    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if y_true else 0.0
    except Exception:
        qwk = 0.0
    return {
        "qwk": qwk,
        "mae": float(mean_absolute_error(y_true, y_pred)) if y_true else 0.0,
        "score_tv": float(tv_distance(true_counts, pred_counts)),
        "score_compression_index": float(pred_std / gold_std) if gold_std > 0 else 0.0,
        "range_coverage": float(len(set(y_pred)) / possible),
        "unique_predicted_scores": int(len(set(y_pred))),
        "possible_score_levels": int(possible),
        "high_score_threshold": int(high_threshold),
        "high_recall": float(high_recall),
        "high_tail_under_score_rate": float(
            sum(1 for i in high_idx if y_pred[i] < high_threshold) / len(high_idx) if high_idx else 0.0
        ),
        "max_recall": float(max_recall),
        "max_score_under_score_rate": float(
            sum(1 for i in max_idx if y_pred[i] < score_max) / len(max_idx) if max_idx else 0.0
        ),
        "prediction_distribution": pred_counts,
        "gold_distribution": true_counts,
    }


def split_val_diag_sel(
    val: Sequence[Dict[str, Any]],
    score_min: int,
    score_max: int,
    diag_ratio: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    by_band: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in sorted(val, key=lambda x: (band_for(int(x["domain1_score"]), score_min, score_max), int(x["domain1_score"]), int(x["essay_id"]))):
        by_band[band_for(int(item["domain1_score"]), score_min, score_max)].append(item)
    diag: List[Dict[str, Any]] = []
    sel: List[Dict[str, Any]] = []
    for band in ["low", "mid", "high"]:
        rows = by_band.get(band, [])
        n_diag = int(round(len(rows) * diag_ratio))
        if len(rows) > 1:
            n_diag = min(max(1, n_diag), len(rows) - 1)
        diag.extend(rows[:n_diag])
        sel.extend(rows[n_diag:])
    diag = sorted(diag, key=lambda x: int(x["essay_id"]))
    sel = sorted(sel, key=lambda x: int(x["essay_id"]))
    meta = {
        "val_diag_ids_hash": id_list_hash([int(x["essay_id"]) for x in diag]),
        "val_sel_ids_hash": id_list_hash([int(x["essay_id"]) for x in sel]),
        "val_diag_n": len(diag),
        "val_sel_n": len(sel),
        "val_diag_score_distribution": score_distribution([int(x["domain1_score"]) for x in diag], score_min, score_max),
        "val_sel_score_distribution": score_distribution([int(x["domain1_score"]) for x in sel], score_min, score_max),
    }
    return diag, sel, meta


def anchor_signature(anchors: Sequence[Dict[str, Any]]) -> str:
    return stable_hash([int(x["essay_id"]) for x in anchors])


def text_jaccard(a: str, b: str) -> float:
    toks_a = tokenize_simple(a)
    toks_b = tokenize_simple(b)
    if not toks_a and not toks_b:
        return 0.0
    return len(toks_a & toks_b) / max(1, len(toks_a | toks_b))


def anchor_metrics(anchors: Sequence[Dict[str, Any]], score_min: int, score_max: int) -> Dict[str, Any]:
    scores = [int(x["gold_score"]) for x in anchors]
    bands = [band_for(score, score_min, score_max) for score in scores]
    band_counts = {band: bands.count(band) for band in ["low", "mid", "high"]}
    pair_scores = []
    pair_rows = []
    for i, left in enumerate(anchors):
        for j in range(i + 1, len(anchors)):
            right = anchors[j]
            sim = text_jaccard(str(left.get("essay_text", "")), str(right.get("essay_text", "")))
            pair_scores.append(sim)
            pair_rows.append((sim, int(left["essay_id"]), int(right["essay_id"])))
    most_pair = max(pair_rows, default=(0.0, None, None), key=lambda x: (x[0], -int(x[1] or 0), -int(x[2] or 0)))
    span = max(scores) - min(scores) if scores else 0
    return {
        "anchor_band_counts": band_counts,
        "anchor_band_coverage": sum(1 for band in ["low", "mid", "high"] if band_counts.get(band, 0) > 0),
        "anchor_unique_score_count": len(set(scores)),
        "anchor_score_range_span": span,
        "anchor_score_range_coverage": float(span / max(1, score_max - score_min)),
        "missing_anchor_bands": [band for band in ["low", "mid", "high"] if band_counts.get(band, 0) == 0],
        "anchor_redundancy_mean": float(np.mean(pair_scores)) if pair_scores else 0.0,
        "anchor_redundancy_max": float(max(pair_scores)) if pair_scores else 0.0,
        "most_redundant_anchor_pair": [most_pair[1], most_pair[2]] if most_pair[1] is not None else [],
    }


def compute_failure_profile(
    y_true_diag: Sequence[int],
    y_pred_diag: Sequence[int],
    anchors: Sequence[Dict[str, Any]],
    score_min: int,
    score_max: int,
    selection_trace: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    metrics = score_metrics(y_true_diag, y_pred_diag, score_min, score_max)
    y_true = [int(x) for x in y_true_diag]
    y_pred = [int(x) for x in y_pred_diag]
    band_mae: Dict[str, Optional[float]] = {}
    band_recall: Dict[str, Optional[float]] = {}
    for band in ["low", "mid", "high"]:
        idx = [i for i, y in enumerate(y_true) if band_for(y, score_min, score_max) == band]
        if not idx:
            band_mae[band] = None
            band_recall[band] = None
            continue
        band_mae[band] = float(mean_absolute_error([y_true[i] for i in idx], [y_pred[i] for i in idx]))
        band_recall[band] = float(
            sum(1 for i in idx if band_for(y_pred[i], score_min, score_max) == band) / len(idx)
        )
    scored_bands = [(band, value) for band, value in band_mae.items() if value is not None]
    worst_band = max(scored_bands, key=lambda x: (float(x[1]), x[0]))[0] if scored_bands else None
    profile = {
        **metrics,
        "band_mae_low": band_mae["low"],
        "band_mae_mid": band_mae["mid"],
        "band_mae_high": band_mae["high"],
        "band_recall_low": band_recall["low"],
        "band_recall_mid": band_recall["mid"],
        "band_recall_high": band_recall["high"],
        "worst_band": worst_band,
        "selection_trace_size": len(selection_trace or []),
    }
    profile.update(anchor_metrics(anchors, score_min, score_max))
    return profile


def rank_repair_operators(failure_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    missing = list(failure_profile.get("missing_anchor_bands") or [])
    if missing or int(failure_profile.get("anchor_band_coverage", 0) or 0) < 3:
        ranked.append(
            {
                "operator": REBALANCE_SCORE_BANDS,
                "severity": float(2.0 + len(missing)),
                "trigger_metrics": {
                    "missing_anchor_bands": missing,
                    "anchor_band_coverage": failure_profile.get("anchor_band_coverage"),
                    "anchor_score_range_coverage": failure_profile.get("anchor_score_range_coverage"),
                },
                "target_band": missing[0] if missing else None,
                "target_score_region": "coverage",
                "target_boundary_metrics": OPERATOR_TARGET_METRICS[REBALANCE_SCORE_BANDS],
                "rationale": "Anchor bank is missing score-band coverage.",
            }
        )
    worst_band = failure_profile.get("worst_band")
    worst_mae = failure_profile.get(f"band_mae_{worst_band}") if worst_band else None
    if worst_band and worst_mae is not None and float(worst_mae) > 0:
        ranked.append(
            {
                "operator": REPLACE_WORST_BAND_ANCHOR,
                "severity": float(worst_mae),
                "trigger_metrics": {
                    "worst_band": worst_band,
                    f"band_mae_{worst_band}": worst_mae,
                    f"band_recall_{worst_band}": failure_profile.get(f"band_recall_{worst_band}"),
                },
                "target_band": worst_band,
                "target_score_region": worst_band,
                "target_boundary_metrics": OPERATOR_TARGET_METRICS[REPLACE_WORST_BAND_ANCHOR],
                "rationale": "Worst validation band has the highest MAE.",
            }
        )
    sci = float(failure_profile.get("score_compression_index", 0.0) or 0.0)
    range_cov = float(failure_profile.get("range_coverage", 0.0) or 0.0)
    high_under = float(failure_profile.get("high_tail_under_score_rate", 0.0) or 0.0)
    max_under = float(failure_profile.get("max_score_under_score_rate", 0.0) or 0.0)
    decompression_severity = max(0.0, 1.0 - sci) + max(0.0, 0.5 - range_cov) + high_under + max_under
    if decompression_severity > 0.25:
        target = "high" if high_under > 0 or max_under > 0 else "low"
        ranked.append(
            {
                "operator": DECOMPRESS_EXTREME_ANCHORS,
                "severity": float(decompression_severity),
                "trigger_metrics": {
                    "score_compression_index": sci,
                    "range_coverage": range_cov,
                    "high_tail_under_score_rate": high_under,
                    "max_score_under_score_rate": max_under,
                },
                "target_band": target,
                "target_score_region": "extreme",
                "target_boundary_metrics": OPERATOR_TARGET_METRICS[DECOMPRESS_EXTREME_ANCHORS],
                "rationale": "Predictions show score compression or tail under-scoring.",
            }
        )
    redundancy_max = float(failure_profile.get("anchor_redundancy_max", 0.0) or 0.0)
    redundancy_mean = float(failure_profile.get("anchor_redundancy_mean", 0.0) or 0.0)
    if redundancy_max >= 0.75 or redundancy_mean >= 0.45:
        ranked.append(
            {
                "operator": REMOVE_CONFUSING_OR_REDUNDANT_ANCHOR,
                "severity": float(redundancy_max + redundancy_mean),
                "trigger_metrics": {
                    "anchor_redundancy_max": redundancy_max,
                    "anchor_redundancy_mean": redundancy_mean,
                    "most_redundant_anchor_pair": failure_profile.get("most_redundant_anchor_pair"),
                },
                "target_band": None,
                "target_score_region": "redundancy",
                "target_boundary_metrics": OPERATOR_TARGET_METRICS[REMOVE_CONFUSING_OR_REDUNDANT_ANCHOR],
                "rationale": "Anchor bank contains redundant or confusing anchors.",
            }
        )
    ranked.sort(key=lambda x: (-float(x["severity"]), str(x["operator"])))
    return ranked


def retrieval_scores(train_pool: Sequence[Dict[str, Any]], val_diag: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    terms = Counter()
    for item in val_diag:
        terms.update(tokenize_simple(str(item.get("essay_text", item.get("essay", "")))))
    scores: Dict[int, float] = {}
    for item in train_pool:
        text = str(item.get("essay_text", item.get("essay", "")))
        toks = tokenize_simple(text)
        scores[int(item["essay_id"])] = float(sum(terms[t] for t in toks) - 0.0005 * token_len(text))
    return scores


def _as_anchor(item: Dict[str, Any], reason: str, selection_score: float = 0.0) -> Dict[str, Any]:
    return {
        "essay_id": int(item["essay_id"]),
        "gold_score": int(item.get("gold_score", item.get("domain1_score"))),
        "prompt_id": int(item.get("prompt_id", 0)),
        "token_length": int(item.get("token_length", token_len(str(item.get("essay_text", item.get("essay", "")))))),
        "source_split": "train",
        "selection_score": float(selection_score),
        "selection_reason": reason,
        "essay_text": str(item.get("essay_text", item.get("essay", ""))),
    }


def _candidate_rows(
    train_pool: Sequence[Dict[str, Any]],
    val_diag: Sequence[Dict[str, Any]],
    anchors: Sequence[Dict[str, Any]],
    score_min: int,
    score_max: int,
    forbidden_ids: Iterable[int],
    target_band: Optional[str] = None,
) -> List[Dict[str, Any]]:
    used = {int(x["essay_id"]) for x in anchors} | {int(x) for x in forbidden_ids}
    scores = retrieval_scores(train_pool, val_diag)
    rows = []
    for item in train_pool:
        essay_id = int(item["essay_id"])
        score = int(item.get("domain1_score", item.get("gold_score")))
        band = band_for(score, score_min, score_max)
        if essay_id in used or (target_band is not None and band != target_band):
            continue
        anchor = _as_anchor(item, "bapr_candidate", scores.get(essay_id, 0.0))
        redundancy = max((text_jaccard(anchor["essay_text"], existing["essay_text"]) for existing in anchors), default=0.0)
        rows.append(
            {
                "anchor": anchor,
                "band": band,
                "retrieval_score": scores.get(essay_id, 0.0),
                "redundancy": redundancy,
                "token_length": anchor["token_length"],
            }
        )
    rows.sort(key=lambda x: (-float(x["retrieval_score"]), float(x["redundancy"]), int(x["token_length"]), int(x["anchor"]["essay_id"])))
    return rows


def _choose_anchor_to_remove(
    anchors: Sequence[Dict[str, Any]],
    operator: str,
    target_band: Optional[str],
    score_min: int,
    score_max: int,
    failure_profile: Dict[str, Any],
) -> Dict[str, Any]:
    if operator == REMOVE_CONFUSING_OR_REDUNDANT_ANCHOR and failure_profile.get("most_redundant_anchor_pair"):
        pair = [int(x) for x in failure_profile["most_redundant_anchor_pair"]]
        pair_anchors = [a for a in anchors if int(a["essay_id"]) in pair]
        if pair_anchors:
            return sorted(pair_anchors, key=lambda a: (band_for(int(a["gold_score"]), score_min, score_max), int(a["gold_score"]), int(a["essay_id"])))[0]
    if operator == DECOMPRESS_EXTREME_ANCHORS:
        mid = [a for a in anchors if band_for(int(a["gold_score"]), score_min, score_max) == "mid"]
        if mid:
            return sorted(mid, key=lambda a: (int(a["gold_score"]), int(a["essay_id"])))[0]
    if target_band:
        same_band = [a for a in anchors if band_for(int(a["gold_score"]), score_min, score_max) == target_band]
        if same_band:
            return sorted(same_band, key=lambda a: (int(a["token_length"]), int(a["essay_id"])), reverse=True)[0]
    counts = Counter(band_for(int(a["gold_score"]), score_min, score_max) for a in anchors)
    return sorted(
        anchors,
        key=lambda a: (-counts[band_for(int(a["gold_score"]), score_min, score_max)], int(a["token_length"]), int(a["essay_id"])),
        reverse=True,
    )[0]


def generate_repaired_children(
    parent_anchors: Sequence[Dict[str, Any]],
    train_pool: Sequence[Dict[str, Any]],
    val_diag: Sequence[Dict[str, Any]],
    failure_profile: Dict[str, Any],
    ranked_operators: Sequence[Dict[str, Any]],
    score_min: int,
    score_max: int,
    k: int,
    forbidden_ids: Iterable[int] = (),
    max_children: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parent = [dict(a) for a in parent_anchors]
    parent_sig = anchor_signature(parent)
    seen = {parent_sig}
    children: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    forbidden = set(int(x) for x in forbidden_ids)
    for op in ranked_operators:
        if len(children) >= max_children:
            break
        operator = str(op["operator"])
        target_band = op.get("target_band")
        if operator == REBALANCE_SCORE_BANDS and failure_profile.get("missing_anchor_bands"):
            target_band = failure_profile["missing_anchor_bands"][0]
        if operator == DECOMPRESS_EXTREME_ANCHORS and not target_band:
            target_band = "high"
        remove = _choose_anchor_to_remove(parent, operator, target_band, score_min, score_max, failure_profile)
        candidates = _candidate_rows(train_pool, val_diag, parent, score_min, score_max, forbidden, target_band)
        if not candidates and target_band is not None:
            candidates = _candidate_rows(train_pool, val_diag, parent, score_min, score_max, forbidden, None)
        if not candidates:
            continue
        added = dict(candidates[0]["anchor"])
        added["selection_reason"] = f"bapr:{operator}"
        child = [dict(a) for a in parent if int(a["essay_id"]) != int(remove["essay_id"])] + [added]
        child = sorted(child, key=lambda a: [int(x["essay_id"]) for x in parent].index(int(a["essay_id"])) if int(a["essay_id"]) in [int(x["essay_id"]) for x in parent] else len(parent))
        if len(child) != k or len({int(a["essay_id"]) for a in child}) != k:
            continue
        sig = anchor_signature(child)
        if sig in seen:
            continue
        seen.add(sig)
        child_id = f"child_{len(children) + 1}"
        bank = {
            "candidate_id": child_id,
            "parent_id": "BAPR-A0",
            "parent_signature": parent_sig,
            "child_signature": sig,
            "operator": operator,
            "severity": float(op.get("severity", 0.0) or 0.0),
            "target_band": target_band,
            "target_boundary_metrics": list(op.get("target_boundary_metrics", OPERATOR_TARGET_METRICS.get(operator, []))),
            "anchors": child,
            "anchor_ids": [int(a["essay_id"]) for a in child],
            "anchor_scores": [int(a["gold_score"]) for a in child],
            "anchor_bank_id": stable_hash({"candidate_id": child_id, "anchor_ids": [int(a["essay_id"]) for a in child], "operator": operator}),
        }
        children.append(bank)
        traces.append(
            {
                "operator": operator,
                "severity": float(op.get("severity", 0.0) or 0.0),
                "trigger_metrics": op.get("trigger_metrics", {}),
                "removed_anchor_id": int(remove["essay_id"]),
                "removed_anchor_score": int(remove["gold_score"]),
                "added_anchor_id": int(added["essay_id"]),
                "added_anchor_score": int(added["gold_score"]),
                "target_band": target_band,
                "candidate_pool_size": len(candidates),
                "replacement_reason": op.get("rationale", ""),
                "parent_anchor_ids": [int(a["essay_id"]) for a in parent],
                "child_anchor_ids": [int(a["essay_id"]) for a in child],
                "child_signature": sig,
            }
        )
    return children, traces


def _metric_improved(parent: Dict[str, Any], child: Dict[str, Any], metric: str) -> bool:
    p = parent.get(metric)
    c = child.get(metric)
    if p is None or c is None:
        return False
    if metric in {"high_recall", "max_recall", "range_coverage"}:
        return float(c) > float(p)
    if metric in {"high_tail_under_score_rate", "max_score_under_score_rate", "score_tv", "worst_band_mae"}:
        return float(c) < float(p)
    if metric == "score_compression_index":
        return abs(float(c) - 1.0) < abs(float(p) - 1.0)
    return False


def boundary_improved_for_operator(parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any], target_metrics: Sequence[str]) -> bool:
    return any(_metric_improved(parent_metrics, child_metrics, metric) for metric in target_metrics)


def guarded_select_repaired_bank(
    parent_metrics: Dict[str, Any],
    child_metrics_list: Sequence[Dict[str, Any]],
    parent_anchor_bank: Dict[str, Any],
    child_anchor_banks: Sequence[Dict[str, Any]],
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = (config or {}).get("bapr", {}).get("guard", config or {})
    qwk_drop = float(cfg.get("qwk_drop_tolerance", 0.02))
    mae_increase = float(cfg.get("mae_increase_tolerance", 0.10))
    rows: List[Dict[str, Any]] = []
    parent_row = {
        "candidate_id": "BAPR-A0",
        "parent_id": "",
        "operator": "PARENT",
        "parent_or_child": "parent",
        "anchor_bank_id": parent_anchor_bank["anchor_bank_id"],
        "anchor_ids": json.dumps(parent_anchor_bank["anchor_ids"]),
        "anchor_scores": json.dumps(parent_anchor_bank["anchor_scores"]),
        "accepted_by_guard": True,
        "guard_reject_reasons": "",
        "target_boundary_metric_improved": False,
        "selected_as_final": False,
        "selected_reason": "",
        **_selection_metric_fields(parent_metrics),
    }
    rows.append(parent_row)
    accepted: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for bank, metrics in zip(child_anchor_banks, child_metrics_list):
        reasons = []
        if float(metrics.get("qwk", 0.0) or 0.0) < float(parent_metrics.get("qwk", 0.0) or 0.0) - qwk_drop:
            reasons.append("qwk_guard")
        if float(metrics.get("mae", 0.0) or 0.0) > float(parent_metrics.get("mae", 0.0) or 0.0) + mae_increase:
            reasons.append("mae_guard")
        for metric in ["anchor_band_coverage", "anchor_unique_score_count"]:
            if int(metrics.get(metric, 0) or 0) < int(parent_metrics.get(metric, 0) or 0):
                reasons.append(metric)
        if int(metrics.get("anchor_score_range_span", 0) or 0) < int(parent_metrics.get("anchor_score_range_span", 0) or 0) - 1:
            reasons.append("anchor_score_range_span")
        target_metrics = bank.get("target_boundary_metrics", OPERATOR_TARGET_METRICS.get(str(bank.get("operator")), []))
        improved = boundary_improved_for_operator(parent_metrics, metrics, target_metrics)
        if not improved:
            reasons.append("target_boundary_metric_not_improved")
        accepted_by_guard = not reasons
        row = {
            "candidate_id": bank["candidate_id"],
            "parent_id": bank.get("parent_id", "BAPR-A0"),
            "operator": bank.get("operator", ""),
            "parent_or_child": "child",
            "anchor_bank_id": bank["anchor_bank_id"],
            "anchor_ids": json.dumps(bank["anchor_ids"]),
            "anchor_scores": json.dumps(bank["anchor_scores"]),
            "accepted_by_guard": accepted_by_guard,
            "guard_reject_reasons": ";".join(reasons),
            "target_boundary_metric_improved": improved,
            "selected_as_final": False,
            "selected_reason": "",
            **_selection_metric_fields(metrics),
        }
        rows.append(row)
        if accepted_by_guard:
            accepted.append((bank, metrics, row))
    if accepted:
        accepted.sort(
            key=lambda item: (
                -float(item[1].get("qwk", 0.0) or 0.0),
                float(item[1].get("mae", 0.0) or 0.0),
                -float(item[1].get("high_recall", 0.0) or 0.0),
                str(item[0].get("candidate_id", "")),
            )
        )
        selected_bank, selected_metrics, _ = accepted[0]
        selected_id = selected_bank["candidate_id"]
        reason = "accepted_child_guarded_selection"
    else:
        selected_bank = parent_anchor_bank
        selected_metrics = parent_metrics
        selected_id = "BAPR-A0"
        reason = "parent_fallback_all_children_rejected"
    for row in rows:
        if row["candidate_id"] == selected_id:
            row["selected_as_final"] = True
            row["selected_reason"] = reason
    return {
        "selected_anchor_bank": selected_bank,
        "selected_metrics": selected_metrics,
        "selection_rows": rows,
        "selected_reason": reason,
    }


def _selection_metric_fields(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "val_sel_qwk": metrics.get("qwk"),
        "val_sel_mae": metrics.get("mae"),
        "val_sel_high_recall": metrics.get("high_recall"),
        "val_sel_high_tail_under_score_rate": metrics.get("high_tail_under_score_rate"),
        "val_sel_max_recall": metrics.get("max_recall"),
        "val_sel_max_score_under_score_rate": metrics.get("max_score_under_score_rate"),
        "val_sel_range_coverage": metrics.get("range_coverage"),
        "val_sel_score_compression_index": metrics.get("score_compression_index"),
        "val_sel_score_tv": metrics.get("score_tv"),
        "val_sel_worst_band_mae": metrics.get("worst_band_mae"),
        "anchor_band_coverage": metrics.get("anchor_band_coverage"),
        "anchor_unique_score_count": metrics.get("anchor_unique_score_count"),
        "anchor_score_range_span": metrics.get("anchor_score_range_span"),
        "token_cost": metrics.get("token_cost"),
    }


def metrics_with_anchor_stats(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    anchors: Sequence[Dict[str, Any]],
    score_min: int,
    score_max: int,
) -> Dict[str, Any]:
    metrics = score_metrics(y_true, y_pred, score_min, score_max)
    band_maes = []
    for band in ["low", "mid", "high"]:
        idx = [i for i, y in enumerate(y_true) if band_for(int(y), score_min, score_max) == band]
        value = float(mean_absolute_error([y_true[i] for i in idx], [y_pred[i] for i in idx])) if idx else None
        metrics[f"band_mae_{band}"] = value
        if value is not None:
            band_maes.append(value)
    metrics["worst_band_mae"] = max(band_maes) if band_maes else None
    metrics.update(anchor_metrics(anchors, score_min, score_max))
    metrics["token_cost"] = sum(int(a.get("token_length", token_len(a.get("essay_text", "")))) for a in anchors)
    return metrics
