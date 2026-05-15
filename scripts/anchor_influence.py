from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from scripts.anchor_stability import stability_by_id
from scripts.bapr_repair import band_for, stable_hash, text_jaccard, token_len


INFLUENCE_REPAIR = "INFLUENCE_REPAIR"


def _score_of(item: Dict[str, Any]) -> int:
    return int(item.get("domain1_score", item.get("gold_score")))


def _text_of(item: Dict[str, Any]) -> str:
    return str(item.get("essay_text", item.get("essay", "")))


def _as_anchor(item: Dict[str, Any], reason: str, selection_score: float = 0.0) -> Dict[str, Any]:
    return {
        "essay_id": int(item["essay_id"]),
        "gold_score": _score_of(item),
        "prompt_id": int(item.get("prompt_id", 0)),
        "token_length": token_len(_text_of(item)),
        "source_split": "train",
        "selection_score": float(selection_score),
        "selection_reason": reason,
        "essay_text": _text_of(item),
    }


def _target_metrics_for_failure(failure_type: str) -> List[str]:
    if failure_type == "high_tail_suppressor":
        return ["high_recall", "high_tail_under_score_rate", "max_recall", "max_score_under_score_rate"]
    if failure_type == "harmful_compression_anchor":
        return ["score_compression_index", "range_coverage", "score_tv"]
    if failure_type == "boundary_confuser":
        return ["worst_band_mae", "score_tv"]
    if failure_type == "redundant_low_influence_anchor":
        return ["score_tv", "anchor_redundancy_mean", "anchor_redundancy_max"]
    return ["anchor_stability_score", "score_tv"]


def _worst_band_from_profile(profile: Dict[str, Any]) -> str | None:
    value = profile.get("worst_band")
    return str(value) if value in {"low", "mid", "high"} else None


def estimate_proxy_influence(
    parent_anchors: Sequence[Dict[str, Any]],
    train_pool: Sequence[Dict[str, Any]],
    val_diag: Sequence[Dict[str, Any]],
    failure_profile: Dict[str, Any],
    stability_rows: Sequence[Dict[str, Any]],
    *,
    score_min: int,
    score_max: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Estimate anchor influence using auditable proxy features.

    The proxy is intentionally label-safe and test-free. It is meant to rank
    anchors for the first BAPR-SI implementation before expensive LOO scoring
    is enabled by the runner.
    """

    del train_pool, val_diag  # reserved for future proxy features
    stability = stability_by_id(stability_rows)
    band_counts = Counter(band_for(int(anchor["gold_score"]), score_min, score_max) for anchor in parent_anchors)
    worst_band = _worst_band_from_profile(failure_profile)
    high_under = float(failure_profile.get("high_tail_under_score_rate", 0.0) or 0.0)
    max_under = float(failure_profile.get("max_score_under_score_rate", 0.0) or 0.0)
    sci = float(failure_profile.get("score_compression_index", 1.0) or 1.0)
    range_coverage = float(failure_profile.get("range_coverage", 1.0) or 1.0)
    worst_band_mae = (
        float(failure_profile.get(f"band_mae_{worst_band}", 0.0) or 0.0)
        if worst_band
        else 0.0
    )

    rows: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    for anchor in parent_anchors:
        essay_id = int(anchor["essay_id"])
        score = int(anchor["gold_score"])
        band = band_for(score, score_min, score_max)
        same_band_count = int(band_counts.get(band, 0))
        other_anchors = [a for a in parent_anchors if int(a["essay_id"]) != essay_id]
        redundancy = max(
            (text_jaccard(str(anchor.get("essay_text", "")), str(other.get("essay_text", ""))) for other in other_anchors),
            default=0.0,
        )
        stab = stability.get(essay_id, {})
        stability_score = float(stab.get("stability_score", 0.0) or 0.0)
        selection_frequency = float(stab.get("selection_frequency", 0.0) or 0.0)
        low_stability = max(0.0, 1.0 - selection_frequency)

        compression_pressure = max(0.0, 0.90 - sci) + max(0.0, 0.45 - range_coverage)
        high_tail_pressure = high_under + max_under
        band_error_pressure = min(1.0, worst_band_mae / max(1.0, float(score_max - score_min))) if worst_band == band else 0.0
        overrep_pressure = max(0.0, float(same_band_count - 3) / 3.0)

        if band == "mid" and compression_pressure > 0.05:
            failure_type = "harmful_compression_anchor"
            pressure = compression_pressure
        elif band == "high" and high_tail_pressure > 0.05 and selection_frequency < 0.75:
            failure_type = "high_tail_suppressor"
            pressure = high_tail_pressure
        elif worst_band == band and band_error_pressure > 0.0:
            failure_type = "boundary_confuser"
            pressure = band_error_pressure
        elif redundancy >= 0.55 or (same_band_count > 3 and selection_frequency < 0.50):
            failure_type = "redundant_low_influence_anchor"
            pressure = max(redundancy, overrep_pressure)
        else:
            failure_type = "useful_stabilizing_anchor"
            pressure = 0.0

        negative_influence_score = (
            0.45 * low_stability
            + 0.25 * float(redundancy)
            + 0.20 * float(pressure)
            + 0.10 * float(overrep_pressure)
            - 0.20 * max(0.0, stability_score)
        )
        if failure_type == "useful_stabilizing_anchor":
            negative_influence_score -= 0.25
        row = {
            "anchor_id": essay_id,
            "essay_id": essay_id,
            "gold_score": score,
            "band": band,
            "anchor_failure_type": failure_type,
            "negative_influence_score": float(negative_influence_score),
            "stability_score": stability_score,
            "selection_frequency": selection_frequency,
            "redundancy_score": float(redundancy),
            "same_band_count": same_band_count,
            "trigger_pressure": float(pressure),
            "target_band": band,
            "target_boundary_metrics": _target_metrics_for_failure(failure_type),
            "expected_repair_metric": _target_metrics_for_failure(failure_type)[0],
            "token_length": int(anchor.get("token_length", token_len(str(anchor.get("essay_text", ""))))),
        }
        rows.append(row)
        trace.append(dict(row))

    rows.sort(
        key=lambda row: (
            -float(row["negative_influence_score"]),
            row["anchor_failure_type"] == "useful_stabilizing_anchor",
            -float(row["redundancy_score"]),
            int(row["essay_id"]),
        )
    )
    return rows, trace


def estimate_leave_one_anchor_out_influence(
    parent_metrics: Dict[str, Any],
    loo_metrics_by_anchor: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare supplied LOO metrics against parent metrics without scoring.

    The runner can later populate ``loo_metrics_by_anchor`` from real scoring;
    this module remains pure logic and never calls the model itself.
    """

    rows = []
    for anchor_id, metrics in loo_metrics_by_anchor.items():
        rows.append(
            {
                "anchor_id": int(anchor_id),
                "delta_qwk_without_anchor": float(metrics.get("qwk", 0.0) or 0.0) - float(parent_metrics.get("qwk", 0.0) or 0.0),
                "delta_mae_without_anchor": float(metrics.get("mae", 0.0) or 0.0) - float(parent_metrics.get("mae", 0.0) or 0.0),
                "delta_high_recall_without_anchor": float(metrics.get("high_recall", 0.0) or 0.0)
                - float(parent_metrics.get("high_recall", 0.0) or 0.0),
                "delta_score_tv_without_anchor": float(metrics.get("score_tv", 0.0) or 0.0)
                - float(parent_metrics.get("score_tv", 0.0) or 0.0),
            }
        )
    return rows


def _candidate_rows(
    train_pool: Sequence[Dict[str, Any]],
    parent_anchors: Sequence[Dict[str, Any]],
    stability_rows: Sequence[Dict[str, Any]],
    *,
    score_min: int,
    score_max: int,
    forbidden_ids: Iterable[int],
    target_band: str,
) -> List[Dict[str, Any]]:
    used = {int(anchor["essay_id"]) for anchor in parent_anchors} | {int(x) for x in forbidden_ids}
    stability = stability_by_id(stability_rows)
    rows: List[Dict[str, Any]] = []
    for item in train_pool:
        essay_id = int(item["essay_id"])
        if essay_id in used:
            continue
        score = _score_of(item)
        band = band_for(score, score_min, score_max)
        if band != target_band:
            continue
        stab = stability.get(essay_id, {})
        stability_score = float(stab.get("stability_score", 0.0) or 0.0)
        redundancy = max(
            (text_jaccard(_text_of(item), str(anchor.get("essay_text", ""))) for anchor in parent_anchors),
            default=0.0,
        )
        combined = stability_score - 0.25 * redundancy - 0.02 * (token_len(_text_of(item)) / 400.0)
        rows.append(
            {
                "anchor": _as_anchor(item, "bapr_si:stable_candidate", stability_score),
                "band": band,
                "stability_score": stability_score,
                "selection_frequency": float(stab.get("selection_frequency", 0.0) or 0.0),
                "redundancy_score": float(redundancy),
                "combined_score": float(combined),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["combined_score"]),
            -float(row["stability_score"]),
            float(row["redundancy_score"]),
            int(row["anchor"]["essay_id"]),
        )
    )
    return rows


def generate_influence_repair_children(
    parent_anchors: Sequence[Dict[str, Any]],
    train_pool: Sequence[Dict[str, Any]],
    val_diag: Sequence[Dict[str, Any]],
    stability_rows: Sequence[Dict[str, Any]],
    influence_rows: Sequence[Dict[str, Any]],
    *,
    score_min: int,
    score_max: int,
    k: int,
    forbidden_ids: Iterable[int] = (),
    max_children: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    del val_diag  # reserved for later LOO influence candidate features
    parent = [dict(anchor) for anchor in parent_anchors]
    parent_ids = [int(anchor["essay_id"]) for anchor in parent]
    parent_sig = stable_hash(parent_ids)
    children: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    seen = {parent_sig}

    for influence in influence_rows:
        if len(children) >= max_children:
            break
        if str(influence.get("anchor_failure_type")) == "useful_stabilizing_anchor":
            continue
        remove_id = int(influence["anchor_id"])
        remove = next((anchor for anchor in parent if int(anchor["essay_id"]) == remove_id), None)
        if remove is None:
            continue
        target_band = str(influence.get("target_band") or band_for(int(remove["gold_score"]), score_min, score_max))
        candidates = _candidate_rows(
            train_pool,
            parent,
            stability_rows,
            score_min=score_min,
            score_max=score_max,
            forbidden_ids=forbidden_ids,
            target_band=target_band,
        )
        if not candidates:
            continue
        added = dict(candidates[0]["anchor"])
        child = [dict(anchor) for anchor in parent if int(anchor["essay_id"]) != remove_id] + [added]
        order = {anchor_id: idx for idx, anchor_id in enumerate(parent_ids)}
        child.sort(key=lambda anchor: order.get(int(anchor["essay_id"]), len(order)))
        if len(child) != k or len({int(anchor["essay_id"]) for anchor in child}) != k:
            continue
        child_sig = stable_hash([int(anchor["essay_id"]) for anchor in child])
        if child_sig in seen:
            continue
        seen.add(child_sig)
        child_id = f"bapr_si_child_{len(children) + 1}"
        target_metrics = list(influence.get("target_boundary_metrics") or [influence.get("expected_repair_metric", "score_tv")])
        bank = {
            "candidate_id": child_id,
            "parent_id": "BAPR-SI-A0",
            "parent_signature": parent_sig,
            "child_signature": child_sig,
            "operator": INFLUENCE_REPAIR,
            "severity": float(influence.get("negative_influence_score", 0.0) or 0.0),
            "target_band": target_band,
            "target_boundary_metrics": target_metrics,
            "anchors": child,
            "anchor_ids": [int(anchor["essay_id"]) for anchor in child],
            "anchor_scores": [int(anchor["gold_score"]) for anchor in child],
            "anchor_bank_id": stable_hash({"candidate_id": child_id, "anchor_ids": [int(anchor["essay_id"]) for anchor in child]}),
            "removed_anchor_id": remove_id,
            "removed_anchor_failure_type": influence.get("anchor_failure_type"),
            "removed_anchor_influence": float(influence.get("negative_influence_score", 0.0) or 0.0),
            "added_anchor_id": int(added["essay_id"]),
            "added_anchor_stability": float(candidates[0]["stability_score"]),
            "expected_repair_metric": influence.get("expected_repair_metric"),
        }
        children.append(bank)
        trace.append(
            {
                "candidate_id": child_id,
                "operator": INFLUENCE_REPAIR,
                "removed_anchor_id": remove_id,
                "removed_anchor_score": int(remove["gold_score"]),
                "removed_anchor_failure_type": influence.get("anchor_failure_type"),
                "removed_anchor_influence": float(influence.get("negative_influence_score", 0.0) or 0.0),
                "added_anchor_id": int(added["essay_id"]),
                "added_anchor_score": int(added["gold_score"]),
                "added_anchor_stability": float(candidates[0]["stability_score"]),
                "added_anchor_selection_frequency": float(candidates[0]["selection_frequency"]),
                "expected_repair_metric": influence.get("expected_repair_metric"),
                "target_boundary_metrics": target_metrics,
                "target_band": target_band,
                "candidate_pool_size": len(candidates),
                "parent_anchor_ids": parent_ids,
                "child_anchor_ids": [int(anchor["essay_id"]) for anchor in child],
                "child_signature": child_sig,
            }
        )
    return children, trace
