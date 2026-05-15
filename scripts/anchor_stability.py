from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from scripts.bapr_repair import band_for, text_jaccard, token_len, tokenize_simple


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
    all_bands = {
        band_for(_score_of(item), score_min, score_max)
        for item in train_pool
    }
    quotas = _band_quota(k, all_bands)

    for b_idx, subset in enumerate(subsets):
        scores = retrieval_scores(train_pool, subset)
        rows = []
        for item in train_pool:
            essay_id = int(item["essay_id"])
            score = _score_of(item)
            band = band_for(score, score_min, score_max)
            rows.append(
                {
                    "item": item,
                    "essay_id": essay_id,
                    "gold_score": score,
                    "band": band,
                    "retrieval_score": float(scores.get(essay_id, 0.0)),
                    "token_length": token_len(_text_of(item)),
                }
            )
        by_band: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_band[str(row["band"])].append(row)
        for band_rows in by_band.values():
            band_rows.sort(key=lambda r: (-float(r["retrieval_score"]), int(r["gold_score"]), int(r["essay_id"])))

        selected_ids: set[int] = set()
        selected_rows: List[Dict[str, Any]] = []
        for band in ["low", "mid", "high"]:
            quota = quotas.get(band, 0)
            for rank, row in enumerate(by_band.get(band, []), start=1):
                row["band_rank"] = rank
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

        for band, band_rows in by_band.items():
            for rank, row in enumerate(band_rows, start=1):
                essay_id = int(row["essay_id"])
                stat = stats.setdefault(
                    essay_id,
                    {
                        "essay_id": essay_id,
                        "gold_score": int(row["gold_score"]),
                        "band": band,
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

        for band in ["low", "mid", "high"]:
            for rank, row in enumerate(by_band.get(band, [])[: max(1, per_band_top_n)], start=1):
                trace.append(
                    {
                        "bootstrap_index": b_idx,
                        "essay_id": int(row["essay_id"]),
                        "gold_score": int(row["gold_score"]),
                        "band": band,
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


def select_stable_anchor_rows(
    train_pool: Sequence[Dict[str, Any]],
    stability_rows: Sequence[Dict[str, Any]],
    *,
    k: int,
    score_min: int,
    score_max: int,
    token_weight: float = 0.02,
    redundancy_weight: float = 0.20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_id = {int(item["essay_id"]): item for item in train_pool}
    rows = [dict(row) for row in stability_rows if int(row["essay_id"]) in by_id]
    if not rows or k <= 0:
        return [], []

    by_band: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_band[str(row["band"])].append(row)
    quotas = _band_quota(k, [band for band, band_rows in by_band.items() if band_rows])
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

    for band in ["low", "mid", "high"]:
        quota = quotas.get(band, 0)
        for _ in range(quota):
            candidates = [row for row in by_band.get(band, []) if int(row["essay_id"]) not in used_ids]
            if not candidates:
                break
            scored = []
            for row in candidates:
                final_score, redundancy, token_penalty = score_row(row, selected)
                scored.append((final_score, redundancy, token_penalty, row))
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
                    "band": band,
                    "score_band": band,
                    "requested_band": band,
                    "band_quota": quota,
                    "selection_frequency": float(best["selection_frequency"]),
                    "mean_rank": float(best["mean_rank"]),
                    "rank_variance": float(best["rank_variance"]),
                    "mean_retrieval_score": float(best["mean_retrieval_score"]),
                    "stability_score": float(best["stability_score"]),
                    "combined_score": float(final_score),
                    "redundancy_score": float(redundancy),
                    "token_cost_penalty": float(token_penalty),
                    "token_length": int(best["token_length"]),
                    "selected_reason": "stability_retrieval: band quota stability selection",
                }
            )
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
                    "requested_band": "fallback_any_band",
                    "band_quota": quotas,
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
