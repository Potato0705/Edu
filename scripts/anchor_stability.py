from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from scripts.bapr_repair import band_for, score_slot_for, score_slot_quota, text_jaccard, token_len, tokenize_simple


def _score_of(item: Dict[str, Any]) -> int:
    return int(item.get("domain1_score", item.get("gold_score")))


def _text_of(item: Dict[str, Any]) -> str:
    return str(item.get("essay_text", item.get("essay", "")))


def _band_quota(k: int, available_bands: Iterable[str]) -> Dict[str, int]:
    bands = [band for band in ["low", "mid", "high"] if band in set(available_bands)]
    if not bands or k <= 0:
        return {}
    base = k // len(bands)
    rem = k % len(bands)
    return {band: base + (1 if i < rem else 0) for i, band in enumerate(bands)}


def retrieval_scores(train_pool: Sequence[Dict[str, Any]], val_diag: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    terms = Counter()
    for item in val_diag:
        terms.update(tokenize_simple(_text_of(item)))
    scores: Dict[int, float] = {}
    for item in train_pool:
        text = _text_of(item)
        toks = tokenize_simple(text)
        scores[int(item["essay_id"])] = float(sum(terms[t] for t in toks) - 0.0005 * token_len(text))
    return scores


def bootstrap_subsets(
    val_diag: Sequence[Dict[str, Any]],
    n_bootstrap: int,
    sample_ratio: float,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    rows = list(val_diag)
    if not rows:
        return [[] for _ in range(max(1, n_bootstrap))]
    rng = random.Random(seed)
    size = max(1, int(round(len(rows) * float(sample_ratio))))
    subsets = []
    for _ in range(max(1, int(n_bootstrap))):
        subsets.append([rows[rng.randrange(len(rows))] for _ in range(size)])
    return subsets


def estimate_anchor_stability(
    train_pool: Sequence[Dict[str, Any]],
    val_diag: Sequence[Dict[str, Any]],
    *,
    k: int,
    score_min: int,
    score_max: int,
    n_bootstrap: int = 8,
    sample_ratio: float = 0.75,
    seed: int = 42,
    per_band_top_n: int = 8,
    rank_variance_weight: float = 0.10,
    redundancy_weight: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Estimate anchor selection stability under diagnostic bootstrap samples.

    This module is intentionally pure logic: it never calls an LLM and only
    consumes train/V_diag text, gold scores, and deterministic retrieval scores.
    """

    if k <= 0 or not train_pool:
        return [], []

    subsets = bootstrap_subsets(val_diag, n_bootstrap, sample_ratio, seed)
    stats: Dict[int, Dict[str, Any]] = {}
    trace: List[Dict[str, Any]] = []
    all_slots = {
        score_slot_for(_score_of(item), score_min, score_max)
        for item in train_pool
    }
    quotas = score_slot_quota(k, all_slots)

    for b_idx, subset in enumerate(subsets):
        scores = retrieval_scores(train_pool, subset)
        rows = []
        for item in train_pool:
            essay_id = int(item["essay_id"])
            score = _score_of(item)
            band = band_for(score, score_min, score_max)
            slot = score_slot_for(score, score_min, score_max)
            rows.append(
                {
                    "item": item,
                    "essay_id": essay_id,
                    "gold_score": score,
                    "band": band,
                    "score_slot": slot,
                    "retrieval_score": float(scores.get(essay_id, 0.0)),
                    "token_length": token_len(_text_of(item)),
                }
            )
        by_slot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_slot[str(row["score_slot"])].append(row)
        for slot_rows in by_slot.values():
            slot_rows.sort(key=lambda r: (-float(r["retrieval_score"]), int(r["gold_score"]), int(r["essay_id"])))

        selected_ids: set[int] = set()
        selected_rows: List[Dict[str, Any]] = []
        for slot in ["low_tail", "lower_mid", "median", "upper_mid", "high_tail", "max_or_top"]:
            quota = quotas.get(slot, 0)
            for rank, row in enumerate(by_slot.get(slot, []), start=1):
                row["slot_rank"] = rank
                if rank <= quota:
                    selected_ids.add(int(row["essay_id"]))
                    selected_rows.append(row)
        if len(selected_ids) < k:
            for row in sorted(rows, key=lambda r: (-float(r["retrieval_score"]), int(r["gold_score"]), int(r["essay_id"]))):
                if int(row["essay_id"]) in selected_ids:
                    continue
                selected_ids.add(int(row["essay_id"]))
                selected_rows.append(row)
                if len(selected_ids) >= k:
                    break

        for slot, slot_rows in by_slot.items():
            for rank, row in enumerate(slot_rows, start=1):
                essay_id = int(row["essay_id"])
                stat = stats.setdefault(
                    essay_id,
                    {
                        "essay_id": essay_id,
                        "gold_score": int(row["gold_score"]),
                        "band": row["band"],
                        "score_slot": slot,
                        "selection_count": 0,
                        "rank_values": [],
                        "retrieval_scores": [],
                        "item": row["item"],
                    },
                )
                if essay_id in selected_ids:
                    stat["selection_count"] += 1
                stat["rank_values"].append(rank)
                stat["retrieval_scores"].append(float(row["retrieval_score"]))

        for slot in ["low_tail", "lower_mid", "median", "upper_mid", "high_tail", "max_or_top"]:
            for rank, row in enumerate(by_slot.get(slot, [])[: max(1, per_band_top_n)], start=1):
                trace.append(
                    {
                        "bootstrap_index": b_idx,
                        "essay_id": int(row["essay_id"]),
                        "gold_score": int(row["gold_score"]),
                        "band": row["band"],
                        "score_slot": slot,
                        "rank": rank,
                        "selected": int(row["essay_id"]) in selected_ids,
                        "retrieval_score": float(row["retrieval_score"]),
                        "bootstrap_val_ids": [int(x["essay_id"]) for x in subset],
                    }
                )

    rows_out: List[Dict[str, Any]] = []
    for stat in stats.values():
        item = stat["item"]
        rank_values = [float(x) for x in stat["rank_values"]]
        selection_frequency = float(stat["selection_count"] / max(1, len(subsets)))
        mean_rank = float(np.mean(rank_values)) if rank_values else float("inf")
        rank_variance = float(np.var(rank_values)) if rank_values else 0.0
        mean_retrieval = float(np.mean(stat["retrieval_scores"])) if stat["retrieval_scores"] else 0.0
        same_band_items = [
            other["item"]
            for other in stats.values()
            if other["essay_id"] != stat["essay_id"] and other["band"] == stat["band"]
        ]
        redundancy = max(
            (text_jaccard(_text_of(item), _text_of(other)) for other in same_band_items),
            default=0.0,
        )
        normalized_rank_variance = rank_variance / max(1.0, float(len(train_pool) ** 2))
        stability_score = (
            selection_frequency
            - float(rank_variance_weight) * normalized_rank_variance
            - float(redundancy_weight) * float(redundancy)
        )
        rows_out.append(
            {
                "essay_id": int(stat["essay_id"]),
                "gold_score": int(stat["gold_score"]),
                "band": stat["band"],
                "selection_frequency": selection_frequency,
                "mean_rank": mean_rank,
                "rank_variance": rank_variance,
                "mean_retrieval_score": mean_retrieval,
                "redundancy_score": float(redundancy),
                "stability_score": float(stability_score),
                "selected_count": int(stat["selection_count"]),
                "bootstrap_count": int(len(subsets)),
                "token_length": token_len(_text_of(item)),
            }
        )
    rows_out.sort(
        key=lambda r: (
            -float(r["stability_score"]),
            -float(r["selection_frequency"]),
            float(r["mean_rank"]),
            int(r["gold_score"]),
            int(r["essay_id"]),
        )
    )
    return rows_out, trace


def stability_by_id(stability_rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(row["essay_id"]): dict(row) for row in stability_rows}


def _uniform_rank_ladder(scores: Sequence[int], k: int) -> List[int]:
    if len(scores) <= k:
        return list(scores)
    if k == 1:
        return [int(scores[len(scores) // 2])]
    n = len(scores)
    indices = []
    for i in range(k):
        idx = int(round(i * (n - 1) / float(k - 1)))
        if idx not in indices:
            indices.append(idx)
    cursor = 0
    while len(indices) < k and cursor < n:
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    indices = sorted(indices[:k])
    return [int(scores[idx]) for idx in indices]


def _upper_tail_dense_ladder(scores: Sequence[int], k: int, score_min: int, score_max: int) -> List[int]:
    if len(scores) <= k:
        return list(scores)
    if k == 1:
        return [int(scores[-1])]
    span = max(1, int(score_max) - int(score_min))
    high_start = int(round(int(score_min) + 0.45 * span))
    selected: set[int] = {int(scores[0]), int(scores[-1])}
    high_scores = [int(score) for score in scores if int(score) >= high_start]
    if len(high_scores) >= k - 1:
        return _uniform_rank_ladder([int(scores[0])] + high_scores, k)
    selected.update(high_scores)
    below = [int(score) for score in scores if int(score) not in selected and int(score) < high_start]
    for score in sorted(below, reverse=True):
        if len(selected) >= k:
            break
        selected.add(score)
    if len(selected) < k:
        for score in scores:
            if len(selected) >= k:
                break
            selected.add(int(score))
    return sorted(selected)[:k]


def target_score_ladder(
    supported_scores: Iterable[int],
    k: int,
    *,
    score_min: int | None = None,
    score_max: int | None = None,
    strategy: str = "uniform_rank",
) -> List[int]:
    scores = sorted({int(score) for score in supported_scores})
    if k <= 0 or not scores:
        return []
    if len(scores) <= k:
        return scores
    strategy_norm = str(strategy or "uniform_rank").lower()
    if strategy_norm == "auto":
        if score_min is not None and score_max is not None and len(scores) <= k + 3:
            return _upper_tail_dense_ladder(scores, k, int(score_min), int(score_max))
        return _uniform_rank_ladder(scores, k)
    if strategy_norm in {"upper_tail_dense", "tail_dense"} and score_min is not None and score_max is not None:
        return _upper_tail_dense_ladder(scores, k, int(score_min), int(score_max))
    return _uniform_rank_ladder(scores, k)


def select_stable_anchor_rows(
    train_pool: Sequence[Dict[str, Any]],
    stability_rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    score_min: int,
    score_max: int,
    token_weight: float = 0.02,
    redundancy_weight: float = 0.20,
    tail_coverage_enabled: bool = False,
    min_top_score_anchors: int = 0,
    top_score_margin: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_id = {int(item["essay_id"]): item for item in train_pool}
    rows = [dict(row) for row in stability_rows if int(row["essay_id"]) in by_id]
    if not rows or k <= 0:
        return [], []

    by_slot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row.setdefault("score_slot", score_slot_for(int(row["gold_score"]), score_min, score_max))
        by_slot[str(row["score_slot"])].append(row)
    quotas = score_slot_quota(k, [slot for slot, slot_rows in by_slot.items() if slot_rows])
    selected: List[Dict[str, Any]] = []
    used_ids: set[int] = set()
    trace: List[Dict[str, Any]] = []

    def score_row(row: Dict[str, Any], current: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
        item = by_id[int(row["essay_id"])]
        redundancy = max(
            (text_jaccard(_text_of(item), _text_of(by_id[int(prev["essay_id"])])) for prev in current),
            default=0.0,
        )
        token_penalty = float(token_weight) * (token_len(_text_of(item)) / 400.0)
        final_score = float(row["stability_score"]) - float(redundancy_weight) * redundancy - token_penalty
        return final_score, float(redundancy), float(token_penalty)

    def select_one(
        candidates: Sequence[Dict[str, Any]],
        requested_slot: str,
        reason: str,
        quota: Any,
        *,
        prefer_high_score: bool = False,
    ) -> bool:
        scored = []
        for row in candidates:
            final_score, redundancy, token_penalty = score_row(row, selected)
            scored.append((final_score, redundancy, token_penalty, row))
        if not scored:
            return False
        if prefer_high_score:
            scored.sort(key=lambda x: (-int(x[3]["gold_score"]), -x[0], x[1], int(x[3]["essay_id"])))
        else:
            scored.sort(key=lambda x: (-x[0], x[1], int(x[3]["essay_id"])))
        final_score, redundancy, token_penalty, best = scored[0]
        selected.append(best)
        used_ids.add(int(best["essay_id"]))
        trace.append(
            {
                "step": len(trace) + 1,
                "essay_id": int(best["essay_id"]),
                "anchor_id": int(best["essay_id"]),
                "gold_score": int(best["gold_score"]),
                "score": int(best["gold_score"]),
                "band": best["band"],
                "score_band": best["band"],
                "score_slot": best["score_slot"],
                "requested_band": best["band"],
                "requested_slot": requested_slot,
                "band_quota": quota,
                "slot_quota": quota,
                "selection_frequency": float(best["selection_frequency"]),
                "mean_rank": float(best["mean_rank"]),
                "rank_variance": float(best["rank_variance"]),
                "mean_retrieval_score": float(best["mean_retrieval_score"]),
                "stability_score": float(best["stability_score"]),
                "combined_score": float(final_score),
                "redundancy_score": float(redundancy),
                "token_cost_penalty": float(token_penalty),
                "token_length": int(best["token_length"]),
                "selected_reason": reason,
            }
        )
        return True

    if tail_coverage_enabled and min_top_score_anchors > 0:
        top_min_score = max(score_min, score_max - max(0, int(top_score_margin)))
        top_candidates = [
            row
            for row in rows
            if int(row["essay_id"]) not in used_ids and int(row["gold_score"]) >= top_min_score
        ]
        for _ in range(max(0, int(min_top_score_anchors))):
            if not top_candidates or len(selected) >= k:
                break
            if not select_one(
                top_candidates,
                requested_slot=f"top_score>={top_min_score}",
                reason="stability_retrieval: top-tail coverage",
                quota={"min_top_score_anchors": int(min_top_score_anchors), "top_score_margin": int(top_score_margin)},
                prefer_high_score=True,
            ):
                break
            top_candidates = [row for row in top_candidates if int(row["essay_id"]) not in used_ids]

    for slot in ["low_tail", "lower_mid", "median", "upper_mid", "high_tail", "max_or_top"]:
        quota = max(0, quotas.get(slot, 0) - sum(1 for row in selected if str(row["score_slot"]) == slot))
        for _ in range(quota):
            candidates = [row for row in by_slot.get(slot, []) if int(row["essay_id"]) not in used_ids]
            if not candidates:
                break
            select_one(candidates, slot, "stability_retrieval: score-slot quota stability selection", quota)
            if len(selected) >= k:
                break
        if len(selected) >= k:
            break

    if len(selected) < k:
        for row in rows:
            if int(row["essay_id"]) in used_ids:
                continue
            final_score, redundancy, token_penalty = score_row(row, selected)
            selected.append(row)
            used_ids.add(int(row["essay_id"]))
            trace.append(
                {
                    "step": len(trace) + 1,
                    "essay_id": int(row["essay_id"]),
                    "anchor_id": int(row["essay_id"]),
                    "gold_score": int(row["gold_score"]),
                    "score": int(row["gold_score"]),
                    "band": row["band"],
                    "score_band": row["band"],
                    "score_slot": row["score_slot"],
                    "requested_band": "fallback_any_band",
                    "requested_slot": "fallback_any_slot",
                    "band_quota": quotas,
                    "slot_quota": quotas,
                    "selection_frequency": float(row["selection_frequency"]),
                    "mean_rank": float(row["mean_rank"]),
                    "rank_variance": float(row["rank_variance"]),
                    "mean_retrieval_score": float(row["mean_retrieval_score"]),
                    "stability_score": float(row["stability_score"]),
                    "combined_score": float(final_score),
                    "redundancy_score": float(redundancy),
                    "token_cost_penalty": float(token_penalty),
                    "token_length": int(row["token_length"]),
                    "selected_reason": "stability_retrieval: fallback fill",
                }
            )
            if len(selected) >= k:
                break
    return selected[:k], trace


def select_coverage_first_sisa_anchor_rows(
    train_pool: Sequence[Dict[str, Any]],
    stability_rows: Sequence[Dict[str, Any]],
    loo_rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    score_min: int,
    score_max: int,
    supported_scores: Iterable[int] | None = None,
    ladder_strategy: str = "uniform_rank",
    candidate_pool_per_score: int = 8,
    retrieval_weight: float = 0.45,
    stability_weight: float = 0.25,
    influence_weight: float = 0.20,
    diversity_weight: float = 0.08,
    token_weight: float = 0.02,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Coverage-first SISA selection.

    The target score ladder is fixed before scoring candidates. Stability and
    LOO influence only choose within the local exact-score or adjacent-score
    candidate pool, so they cannot break score-scale coverage.
    """

    by_id = {int(item["essay_id"]): item for item in train_pool}
    rows = [dict(row) for row in stability_rows if int(row["essay_id"]) in by_id]
    if not rows or k <= 0:
        return [], []

    if supported_scores is None:
        supported_scores = [int(row["gold_score"]) for row in rows]
    ladder = target_score_ladder(supported_scores, k, score_min=score_min, score_max=score_max, strategy=ladder_strategy)
    if not ladder:
        return [], []

    loo_by_id = {int(row["anchor_id"]): row for row in loo_rows if "anchor_id" in row}
    max_abs_retrieval = max((abs(float(row.get("mean_retrieval_score", 0.0) or 0.0)) for row in rows), default=1.0) or 1.0

    rows_by_score: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        essay_id = int(row["essay_id"])
        score = int(row["gold_score"])
        row["score_slot"] = score_slot_for(score, score_min, score_max)
        row["band"] = row.get("band") or band_for(score, score_min, score_max)
        loo = loo_by_id.get(essay_id, {})
        delta_qwk = float(loo.get("delta_qwk_without_anchor", 0.0) or 0.0) if loo else 0.0
        delta_mae = float(loo.get("delta_mae_without_anchor", 0.0) or 0.0) if loo else 0.0
        row["loo_available"] = bool(loo)
        row["loo_delta_qwk_without_anchor"] = delta_qwk if loo else None
        row["loo_delta_mae_without_anchor"] = delta_mae if loo else None
        row["loo_influence_score"] = float(
            max(0.0, -delta_qwk)
            + max(0.0, delta_mae)
            - max(0.0, delta_qwk)
            - max(0.0, -delta_mae)
        )
        row["retrieval_score_normalized"] = float(row.get("mean_retrieval_score", 0.0) or 0.0) / max_abs_retrieval
        rows_by_score[score].append(row)

    selected: List[Dict[str, Any]] = []
    used_ids: set[int] = set()
    trace: List[Dict[str, Any]] = []

    def coverage_candidates(target_score: int) -> Tuple[List[Dict[str, Any]], bool, int, str]:
        exact = [row for row in rows_by_score.get(target_score, []) if int(row["essay_id"]) not in used_ids]
        if exact:
            return exact, True, 0, ""
        available_scores = sorted(score for score, score_rows in rows_by_score.items() if any(int(row["essay_id"]) not in used_ids for row in score_rows))
        if not available_scores:
            return [], False, 0, "no_available_candidates"
        nearest = min(available_scores, key=lambda score: (abs(score - target_score), -score if target_score >= score_max else score))
        fallback = [row for row in rows_by_score.get(nearest, []) if int(row["essay_id"]) not in used_ids]
        return fallback, False, int(nearest - target_score), "adjacent_score_fallback"

    def score_candidate(row: Dict[str, Any], current: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        item = by_id[int(row["essay_id"])]
        redundancy = max(
            (text_jaccard(_text_of(item), _text_of(by_id[int(prev["essay_id"])])) for prev in current),
            default=0.0,
        )
        token_penalty = float(token_weight) * (token_len(_text_of(item)) / 400.0)
        retrieval_component = float(retrieval_weight) * float(row.get("retrieval_score_normalized", 0.0) or 0.0)
        stability_component = float(stability_weight) * float(row.get("stability_score", 0.0) or 0.0)
        influence_component = float(influence_weight) * float(row.get("loo_influence_score", 0.0) or 0.0)
        diversity_penalty = float(diversity_weight) * redundancy
        combined = retrieval_component + stability_component + influence_component - diversity_penalty - token_penalty
        return {
            "combined_score": float(combined),
            "retrieval_component": float(retrieval_component),
            "stability_component": float(stability_component),
            "influence_component": float(influence_component),
            "diversity_penalty": float(diversity_penalty),
            "redundancy_score": float(redundancy),
            "token_cost_penalty": float(token_penalty),
        }

    for target_score in ladder:
        if len(selected) >= k:
            break
        candidates, exact_match, coverage_gap, fallback_reason = coverage_candidates(int(target_score))
        if not candidates:
            trace.append(
                {
                    "step": len(trace) + 1,
                    "target_score": int(target_score),
                    "target_score_ladder": ladder,
                    "selected": False,
                    "selected_reason": "coverage_first_sisa: no candidate available",
                    "fallback_reason": fallback_reason,
                }
            )
            continue
        local_pool = sorted(
            candidates,
            key=lambda row: (
                -float(row.get("retrieval_score_normalized", 0.0) or 0.0),
                -float(row.get("stability_score", 0.0) or 0.0),
                int(row["essay_id"]),
            ),
        )[: max(1, int(candidate_pool_per_score))]
        scored = []
        for row in local_pool:
            parts = score_candidate(row, selected)
            scored.append((parts["combined_score"], parts["redundancy_score"], int(row["essay_id"]), row, parts))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        _, _, _, best, parts = scored[0]
        selected.append(best)
        used_ids.add(int(best["essay_id"]))
        local_retrieval_top = sorted(
            local_pool,
            key=lambda row: (-float(row.get("retrieval_score_normalized", 0.0) or 0.0), int(row["essay_id"])),
        )[0]
        stability_or_influence_changed_choice = int(local_retrieval_top["essay_id"]) != int(best["essay_id"])
        trace.append(
            {
                "step": len(trace) + 1,
                "essay_id": int(best["essay_id"]),
                "anchor_id": int(best["essay_id"]),
                "gold_score": int(best["gold_score"]),
                "score": int(best["gold_score"]),
                "band": best["band"],
                "score_band": best["band"],
                "score_slot": best["score_slot"],
                "target_score": int(target_score),
                "target_score_ladder": ladder,
                "requested_score": int(target_score),
                "exact_score_match": bool(exact_match),
                "coverage_gap": int(coverage_gap),
                "fallback_reason": fallback_reason,
                "candidate_pool_size": int(len(local_pool)),
                "candidate_pool_per_score": int(candidate_pool_per_score),
                "retrieval_top_essay_id": int(local_retrieval_top["essay_id"]),
                "stability_or_influence_changed_local_choice": bool(stability_or_influence_changed_choice),
                "selection_frequency": float(best.get("selection_frequency", 0.0) or 0.0),
                "mean_rank": float(best.get("mean_rank", 0.0) or 0.0),
                "rank_variance": float(best.get("rank_variance", 0.0) or 0.0),
                "mean_retrieval_score": float(best.get("mean_retrieval_score", 0.0) or 0.0),
                "retrieval_score_normalized": float(best.get("retrieval_score_normalized", 0.0) or 0.0),
                "stability_score": float(best.get("stability_score", 0.0) or 0.0),
                "loo_available": bool(best.get("loo_available", False)),
                "loo_influence_score": float(best.get("loo_influence_score", 0.0) or 0.0),
                "loo_delta_qwk_without_anchor": best.get("loo_delta_qwk_without_anchor"),
                "loo_delta_mae_without_anchor": best.get("loo_delta_mae_without_anchor"),
                **parts,
                "token_length": int(best["token_length"]),
                "selected_reason": "coverage_first_sisa: target-score local stability-influence selection",
                "selection_parts": {
                    "retrieval_weight": float(retrieval_weight),
                    "stability_weight": float(stability_weight),
                    "influence_weight": float(influence_weight),
                    "diversity_weight": float(diversity_weight),
                    "token_weight": float(token_weight),
                },
            }
        )

    return selected[:k], trace


def select_sisa_anchor_rows(
    train_pool: Sequence[Dict[str, Any]],
    stability_rows: Sequence[Dict[str, Any]],
    loo_rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    score_min: int,
    score_max: int,
    stability_weight: float = 0.45,
    retrieval_weight: float = 0.25,
    influence_weight: float = 0.20,
    redundancy_weight: float = 0.20,
    token_weight: float = 0.02,
    tail_coverage_enabled: bool = True,
    min_top_score_anchors: int = 1,
    top_score_margin: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select scale-aware anchors using stability plus LOO influence signals.

    SISA is parent-only: influence acts as an anchor-level prior during subset
    selection, not as a post-hoc repair operator. LOO evidence is available only
    for the initial stable parent anchors; other candidates receive a neutral
    influence score.
    """

    by_id = {int(item["essay_id"]): item for item in train_pool}
    rows = [dict(row) for row in stability_rows if int(row["essay_id"]) in by_id]
    if not rows or k <= 0:
        return [], []
    loo_by_id = {int(row["anchor_id"]): row for row in loo_rows if "anchor_id" in row}
    for row in rows:
        essay_id = int(row["essay_id"])
        row.setdefault("score_slot", score_slot_for(int(row["gold_score"]), score_min, score_max))
        loo = loo_by_id.get(essay_id, {})
        delta_qwk = float(loo.get("delta_qwk_without_anchor", 0.0) or 0.0) if loo else 0.0
        delta_mae = float(loo.get("delta_mae_without_anchor", 0.0) or 0.0) if loo else 0.0
        # Positive means removing the anchor hurts; negative means removal helps.
        influence_score = max(0.0, -delta_qwk) + max(0.0, delta_mae) - max(0.0, delta_qwk) - max(0.0, -delta_mae)
        row["loo_available"] = bool(loo)
        row["loo_delta_qwk_without_anchor"] = delta_qwk if loo else None
        row["loo_delta_mae_without_anchor"] = delta_mae if loo else None
        row["loo_influence_score"] = float(influence_score)

    by_slot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_slot[str(row["score_slot"])].append(row)
    quotas = score_slot_quota(k, [slot for slot, slot_rows in by_slot.items() if slot_rows])
    selected: List[Dict[str, Any]] = []
    used_ids: set[int] = set()
    trace: List[Dict[str, Any]] = []

    max_retrieval = max((abs(float(row.get("mean_retrieval_score", 0.0) or 0.0)) for row in rows), default=1.0) or 1.0

    def score_row(row: Dict[str, Any], current: Sequence[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        item = by_id[int(row["essay_id"])]
        redundancy = max(
            (text_jaccard(_text_of(item), _text_of(by_id[int(prev["essay_id"])])) for prev in current),
            default=0.0,
        )
        token_penalty = float(token_weight) * (token_len(_text_of(item)) / 400.0)
        retrieval_norm = float(row.get("mean_retrieval_score", 0.0) or 0.0) / max_retrieval
        final_score = (
            float(stability_weight) * float(row.get("stability_score", 0.0) or 0.0)
            + float(retrieval_weight) * retrieval_norm
            + float(influence_weight) * float(row.get("loo_influence_score", 0.0) or 0.0)
            - float(redundancy_weight) * redundancy
            - token_penalty
        )
        return float(final_score), float(redundancy), float(token_penalty), float(retrieval_norm)

    def select_one(candidates: Sequence[Dict[str, Any]], requested_slot: str, reason: str, quota: Any, prefer_high_score: bool = False) -> bool:
        scored = []
        for row in candidates:
            final_score, redundancy, token_penalty, retrieval_norm = score_row(row, selected)
            scored.append((final_score, redundancy, token_penalty, retrieval_norm, row))
        if not scored:
            return False
        if prefer_high_score:
            scored.sort(key=lambda x: (-int(x[4]["gold_score"]), -x[0], x[1], int(x[4]["essay_id"])))
        else:
            scored.sort(key=lambda x: (-x[0], x[1], int(x[4]["essay_id"])))
        final_score, redundancy, token_penalty, retrieval_norm, best = scored[0]
        selected.append(best)
        used_ids.add(int(best["essay_id"]))
        trace.append(
            {
                "step": len(trace) + 1,
                "essay_id": int(best["essay_id"]),
                "anchor_id": int(best["essay_id"]),
                "gold_score": int(best["gold_score"]),
                "score": int(best["gold_score"]),
                "band": best["band"],
                "score_band": best["band"],
                "score_slot": best["score_slot"],
                "requested_band": best["band"],
                "requested_slot": requested_slot,
                "slot_quota": quota,
                "selection_frequency": float(best.get("selection_frequency", 0.0) or 0.0),
                "mean_rank": float(best.get("mean_rank", 0.0) or 0.0),
                "rank_variance": float(best.get("rank_variance", 0.0) or 0.0),
                "mean_retrieval_score": float(best.get("mean_retrieval_score", 0.0) or 0.0),
                "retrieval_score_normalized": retrieval_norm,
                "stability_score": float(best.get("stability_score", 0.0) or 0.0),
                "loo_available": bool(best.get("loo_available", False)),
                "loo_influence_score": float(best.get("loo_influence_score", 0.0) or 0.0),
                "loo_delta_qwk_without_anchor": best.get("loo_delta_qwk_without_anchor"),
                "loo_delta_mae_without_anchor": best.get("loo_delta_mae_without_anchor"),
                "combined_score": float(final_score),
                "redundancy_score": float(redundancy),
                "token_cost_penalty": float(token_penalty),
                "token_length": int(best["token_length"]),
                "selected_reason": reason,
                "selection_parts": {
                    "stability_weight": float(stability_weight),
                    "retrieval_weight": float(retrieval_weight),
                    "influence_weight": float(influence_weight),
                    "redundancy_weight": float(redundancy_weight),
                    "token_weight": float(token_weight),
                },
            }
        )
        return True

    if tail_coverage_enabled and min_top_score_anchors > 0:
        top_min_score = max(score_min, score_max - max(0, int(top_score_margin)))
        top_candidates = [row for row in rows if int(row["essay_id"]) not in used_ids and int(row["gold_score"]) >= top_min_score]
        for _ in range(max(0, int(min_top_score_anchors))):
            if not top_candidates or len(selected) >= k:
                break
            if not select_one(
                top_candidates,
                requested_slot=f"top_score>={top_min_score}",
                reason="sisa: protected top-tail scale anchor",
                quota={"min_top_score_anchors": int(min_top_score_anchors), "top_score_margin": int(top_score_margin)},
                prefer_high_score=True,
            ):
                break
            top_candidates = [row for row in top_candidates if int(row["essay_id"]) not in used_ids]

    for slot in ["low_tail", "lower_mid", "median", "upper_mid", "high_tail", "max_or_top"]:
        quota = max(0, quotas.get(slot, 0) - sum(1 for row in selected if str(row["score_slot"]) == slot))
        for _ in range(quota):
            candidates = [row for row in by_slot.get(slot, []) if int(row["essay_id"]) not in used_ids]
            if not candidates:
                break
            select_one(candidates, slot, "sisa: score-slot stability-influence selection", quota)
            if len(selected) >= k:
                break
        if len(selected) >= k:
            break

    if len(selected) < k:
        for row in sorted(rows, key=lambda r: (-float(r.get("stability_score", 0.0) or 0.0), int(r["essay_id"]))):
            if int(row["essay_id"]) in used_ids:
                continue
            select_one([row], "fallback_any_slot", "sisa: fallback fill", quotas)
            if len(selected) >= k:
                break
    return selected[:k], trace
