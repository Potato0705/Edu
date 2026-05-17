from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.llm_backend import LocalLlamaBackend, ScoringRequest  # noqa: E402
from scripts.bapr_repair import (  # noqa: E402
    compute_failure_profile,
    generate_repaired_children,
    guarded_select_repaired_bank,
    metrics_with_anchor_stats,
    rank_repair_operators,
    score_slot_for as bapr_score_slot_for,
    score_slot_quota,
    split_val_diag_sel,
    stable_hash as bapr_stable_hash,
)
from scripts.anchor_influence import (  # noqa: E402
    apply_loo_veto_to_proxy_influence,
    estimate_leave_one_anchor_out_influence,
    estimate_proxy_influence,
    generate_influence_repair_children,
)
from scripts.anchor_stability import (  # noqa: E402
    estimate_anchor_stability,
    select_coverage_first_sisa_anchor_rows,
    select_sisa_anchor_rows,
    select_stable_anchor_rows,
    stability_by_id,
)
from wise_aes import (  # noqa: E402
    PromptIndividual,
    _adaptive_high_score_threshold,
    _max_score_contract_text,
    _score_band_label,
    _score_band_labels,
    _stratified_debug_split,
)


@dataclass
class AnchorRecord:
    essay_id: int
    gold_score: int
    prompt_id: int
    token_length: int
    source_split: str
    selection_score: float
    selection_reason: str
    essay_text: str

    def to_prompt_example(self) -> Dict[str, Any]:
        return {
            "essay_id": self.essay_id,
            "essay_text": self.essay_text,
            "domain1_score": self.gold_score,
        }


@dataclass
class AnchorBank:
    anchor_bank_id: str
    method: str
    k: int
    anchor_ids: List[int]
    score_coverage: Dict[str, int]
    token_cost: int
    score_range: List[int]
    selection_trace_path: str
    representation_changed_anchor_choice: bool = False
    representation_features_used: Optional[List[str]] = None


def stable_hash(payload: Any, n: int = 12) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def token_len(text: str) -> int:
    return max(1, len(str(text).split()))


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


def score_boundary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    score_min: int,
    score_max: int,
) -> Dict[str, Any]:
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]
    true_counts = score_distribution(y_true, score_min, score_max)
    pred_counts = score_distribution(y_pred, score_min, score_max)
    high_threshold = _adaptive_high_score_threshold(
        {"data": {"score_min": score_min, "score_max": score_max}},
        list(y_true),
    )
    high_idx = [i for i, y in enumerate(y_true) if y >= high_threshold]
    max_idx = [i for i, y in enumerate(y_true) if y == score_max]
    high_recall = (
        sum(1 for i in high_idx if y_pred[i] >= high_threshold) / len(high_idx)
        if high_idx
        else 1.0
    )
    max_recall = (
        sum(1 for i in max_idx if y_pred[i] == score_max) / len(max_idx)
        if max_idx
        else 1.0
    )
    per_score_recall = {}
    for score in range(score_min, score_max + 1):
        idx = [i for i, y in enumerate(y_true) if y == score]
        per_score_recall[str(score)] = (
            sum(1 for i in idx if y_pred[i] == score) / len(idx) if idx else None
        )
    gold_std = float(np.std(y_true)) if y_true else 0.0
    pred_std = float(np.std(y_pred)) if y_pred else 0.0
    possible = max(1, score_max - score_min + 1)
    high_under = (
        sum(1 for i in high_idx if y_pred[i] < high_threshold) / len(high_idx)
        if high_idx
        else 0.0
    )
    max_under = (
        sum(1 for i in max_idx if y_pred[i] < score_max) / len(max_idx)
        if max_idx
        else 0.0
    )
    return {
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if y_true else 0.0,
        "mae": float(mean_absolute_error(y_true, y_pred)) if y_true else 0.0,
        "high_score_threshold": int(high_threshold),
        "high_recall": float(high_recall),
        "max_recall": float(max_recall),
        "score_compression_index": float(pred_std / gold_std) if gold_std > 0 else 0.0,
        "range_coverage": float(len(set(y_pred)) / possible),
        "unique_predicted_scores": int(len(set(y_pred))),
        "possible_score_levels": int(possible),
        "per_score_recall": per_score_recall,
        "prediction_distribution": pred_counts,
        "gold_distribution": true_counts,
        "score_tv": float(tv_distance(true_counts, pred_counts)),
        "high_tail_under_score_rate": float(high_under),
        "max_score_under_score_rate": float(max_under),
    }


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def id_list_hash(ids: Sequence[int]) -> str:
    return hashlib.md5(",".join(map(str, ids)).encode("utf-8")).hexdigest()[:16]


def load_asap_data(config: Dict[str, Any]) -> List[Dict]:
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["asap_path"], sep="\t", encoding="latin-1")
    df = df[df["essay_set"] == int(data_cfg["essay_set"])]
    return [
        {
            "essay_id": int(row["essay_id"]),
            "essay_text": str(row["essay"]),
            "domain1_score": int(row["domain1_score"]),
        }
        for _, row in df.iterrows()
    ]


def load_split_manifest(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    for key in ["train_ids", "val_ids", "test_ids"]:
        if key not in manifest:
            raise ValueError(f"Split manifest missing required key: {key}")
        manifest[key] = [int(x) for x in manifest[key]]
    manifest["anchor_pool_ids"] = [int(x) for x in manifest.get("anchor_pool_ids", manifest["train_ids"])]
    return manifest


def split_by_manifest(config: Dict[str, Any], manifest: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    all_data = load_asap_data(config)
    by_id = {int(x["essay_id"]): x for x in all_data}
    missing = []
    splits = []
    for key in ["train_ids", "val_ids", "test_ids"]:
        rows = []
        for essay_id in manifest[key]:
            item = by_id.get(int(essay_id))
            if item is None:
                missing.append(int(essay_id))
            else:
                rows.append(item)
        splits.append(rows)
    if missing:
        raise ValueError(f"Split manifest contains essay IDs not found in configured ASAP prompt: {missing[:10]}")
    test_ids = set(manifest["test_ids"])
    leaked = sorted(set(manifest.get("anchor_pool_ids", manifest["train_ids"])) & test_ids)
    if leaked:
        raise ValueError(f"Split manifest anchor_pool_ids overlap test_ids: {leaked[:10]}")
    return splits[0], splits[1], splits[2]


def split_hash_summary(train: Sequence[Dict], val: Sequence[Dict], test: Sequence[Dict]) -> Dict[str, Any]:
    train_ids = [int(x["essay_id"]) for x in train]
    val_ids = [int(x["essay_id"]) for x in val]
    test_ids = [int(x["essay_id"]) for x in test]
    return {
        "train_n": len(train_ids),
        "val_n": len(val_ids),
        "test_n": len(test_ids),
        "train_ids_hash": id_list_hash(train_ids),
        "val_ids_hash": id_list_hash(val_ids),
        "test_ids_hash": id_list_hash(test_ids),
        "anchor_pool_ids_hash": id_list_hash(train_ids),
    }


def load_asap_split(config: Dict[str, Any], fold: int, split_manifest: Optional[Path] = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if split_manifest is not None:
        return split_by_manifest(config, load_split_manifest(split_manifest))
    all_data = load_asap_data(config)
    data_cfg = config["data"]
    score_min = int(data_cfg["score_min"])
    score_max = int(data_cfg["score_max"])
    dbg = config.get("debug", {})
    seed = int(dbg.get("seed", 42 + fold))
    if dbg.get("enabled", False):
        if dbg.get("stratified", True):
            return _stratified_debug_split(
                all_data,
                int(dbg.get("n_train", 240)),
                int(dbg.get("n_val", 96)),
                int(dbg.get("n_test", 64)),
                seed,
                score_min,
                score_max,
            )
        random.Random(seed).shuffle(all_data)
        n_train = int(dbg.get("n_train", 240))
        n_val = int(dbg.get("n_val", 96))
        n_test = int(dbg.get("n_test", 64))
        return all_data[:n_train], all_data[n_train:n_train + n_val], all_data[n_train + n_val:n_train + n_val + n_test]

    labels = _score_band_labels(all_data, score_min, score_max)
    try:
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(all_data, labels))
    except Exception:
        folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(all_data))
    train_val_idx, test_idx = folds[fold]
    train_val = [all_data[i] for i in train_val_idx]
    test = [all_data[i] for i in test_idx]
    train_val_labels = _score_band_labels(train_val, score_min, score_max)
    try:
        train, val = train_test_split(
            train_val,
            test_size=0.2,
            stratify=train_val_labels,
            random_state=42 + fold,
        )
        return list(train), list(val), test
    except Exception:
        split = int(len(train_val) * 0.8)
        return train_val[:split], train_val[split:], test


def instruction_from_config(config: Dict[str, Any]) -> str:
    official = config.get("induction", {}).get(
        "official_criteria",
        "Evaluate the essay based on content, organization, and language use.",
    )
    score_min = int(config["data"]["score_min"])
    score_max = int(config["data"]["score_max"])
    contract = (
        f"Score Range Contract: Use only integer final_score values from {score_min} to {score_max}, inclusive. "
        "Do not use incompatible point totals, percentages, letter grades, or decimal scores."
    )
    instruction = f"{contract}\n\nScoring Rubric:\n{official}"
    max_contract = _max_score_contract_text(config)
    if max_contract:
        instruction += f"\n\nMaximum-Score Attainable Contract:\n{max_contract}"
    return instruction


def score_coverage(anchors: Sequence[AnchorRecord], score_min: int, score_max: int) -> Dict[str, int]:
    counts = {str(s): 0 for s in range(score_min, score_max + 1)}
    for anchor in anchors:
        counts[str(anchor.gold_score)] = counts.get(str(anchor.gold_score), 0) + 1
    return counts


def band_for(score: int, score_min: int, score_max: int) -> str:
    return _score_band_label(score, score_min, score_max)


def score_slot_for(score: int, score_min: int, score_max: int) -> str:
    return bapr_score_slot_for(score, score_min, score_max)


def deterministic_score_covered(
    train: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    seed: int,
    method: str,
) -> List[AnchorRecord]:
    by_score: Dict[int, List[Dict]] = defaultdict(list)
    for item in sorted(train, key=lambda x: (int(x["domain1_score"]), int(x["essay_id"]))):
        by_score[int(item["domain1_score"])].append(item)
    available_scores = [s for s in range(score_min, score_max + 1) if by_score.get(s)]
    if not available_scores or k <= 0:
        return []
    selected: List[Dict] = []
    offset = seed % len(available_scores)
    scores = available_scores[offset:] + available_scores[:offset]
    cursor = {s: 0 for s in scores}
    while len(selected) < k and len(selected) < len(train):
        progressed = False
        for score in scores:
            pool = by_score[score]
            if cursor[score] < len(pool):
                selected.append(pool[cursor[score]])
                cursor[score] += 1
                progressed = True
                if len(selected) >= k:
                    break
        if not progressed:
            break
    return [
        AnchorRecord(
            essay_id=int(x["essay_id"]),
            gold_score=int(x["domain1_score"]),
            prompt_id=0,
            token_length=token_len(x["essay_text"]),
            source_split="train",
            selection_score=1.0,
            selection_reason=f"{method}: deterministic score coverage",
            essay_text=x["essay_text"],
        )
        for x in selected
    ]


def stratified_anchors(
    train: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    seed: int,
) -> List[AnchorRecord]:
    by_band: Dict[str, List[Dict]] = defaultdict(list)
    for item in sorted(train, key=lambda x: (band_for(int(x["domain1_score"]), score_min, score_max), int(x["domain1_score"]), int(x["essay_id"]))):
        by_band[band_for(int(item["domain1_score"]), score_min, score_max)].append(item)
    bands = ["low", "mid", "high"]
    selected: List[Dict] = []
    cursor = {b: seed % max(1, len(by_band[b])) if by_band[b] else 0 for b in bands}
    while len(selected) < k:
        progressed = False
        for band in bands:
            pool = by_band.get(band, [])
            if not pool:
                continue
            idx = cursor[band] % len(pool)
            item = pool[idx]
            cursor[band] += 1
            if item["essay_id"] not in {x["essay_id"] for x in selected}:
                selected.append(item)
                progressed = True
                if len(selected) >= k:
                    break
        if not progressed:
            break
    return [
        AnchorRecord(
            essay_id=int(x["essay_id"]),
            gold_score=int(x["domain1_score"]),
            prompt_id=0,
            token_length=token_len(x["essay_text"]),
            source_split="train",
            selection_score=1.0,
            selection_reason="stratified: low/mid/high score-band coverage",
            essay_text=x["essay_text"],
        )
        for x in selected
    ]


def tokenize_simple(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{3,}", str(text).lower()))


def retrieval_anchors(train: Sequence[Dict], val: Sequence[Dict], k: int, score_min: int, score_max: int) -> List[AnchorRecord]:
    val_terms = Counter()
    for item in val:
        val_terms.update(tokenize_simple(item["essay_text"]))
    scored = []
    for item in train:
        terms = tokenize_simple(item["essay_text"])
        overlap = sum(val_terms[t] for t in terms)
        length_pen = 0.0005 * token_len(item["essay_text"])
        scored.append((overlap - length_pen, item))
    scored.sort(key=lambda x: (-x[0], int(x[1]["domain1_score"]), int(x[1]["essay_id"])))
    by_slot: Dict[str, List[Tuple[float, Dict]]] = defaultdict(list)
    for score, item in scored:
        by_slot[score_slot_for(int(item["domain1_score"]), score_min, score_max)].append((score, item))
    quotas = score_slot_quota(k, [slot for slot, rows in by_slot.items() if rows])
    selected: List[Dict] = []
    used_ids: set[int] = set()
    for slot in ["low_tail", "lower_mid", "median", "upper_mid", "high_tail", "max_or_top"]:
        for score, item in by_slot.get(slot, [])[: quotas.get(slot, 0)]:
            selected.append(item)
            used_ids.add(int(item["essay_id"]))
    for _, item in scored:
        if len(selected) >= k:
            break
        if int(item["essay_id"]) not in used_ids:
            selected.append(item)
            used_ids.add(int(item["essay_id"]))
    return [
        AnchorRecord(
            essay_id=int(x["essay_id"]),
            gold_score=int(x["domain1_score"]),
            prompt_id=0,
            token_length=token_len(x["essay_text"]),
            source_split="train",
            selection_score=float(next((s for s, y in scored if y["essay_id"] == x["essay_id"]), 0.0)),
            selection_reason="retrieval: lexical validation-pool similarity with score coverage",
            essay_text=x["essay_text"],
        )
        for x in selected[:k]
    ]


def tfidf_vectors(items: Sequence[Dict]) -> Tuple[np.ndarray, List[str]]:
    vocab: Dict[str, int] = {}
    docs = []
    for item in items:
        toks = sorted(tokenize_simple(item["essay_text"]))
        docs.append(toks)
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    mat = np.zeros((len(items), max(1, len(vocab))), dtype=float)
    df = Counter(tok for toks in docs for tok in set(toks))
    for i, toks in enumerate(docs):
        counts = Counter(toks)
        for tok, cnt in counts.items():
            mat[i, vocab[tok]] = cnt * math.log((1 + len(items)) / (1 + df[tok])) + 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.maximum(norms, 1e-8)
    inv_vocab = [None] * len(vocab)
    for tok, idx in vocab.items():
        inv_vocab[idx] = tok
    return mat, inv_vocab  # type: ignore[return-value]


def representation_guided_anchors(
    train: Sequence[Dict],
    val: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]]]:
    rep_cfg = config.get("anchor_budget", {}).get("representation", {})
    mode = str(rep_cfg.get("mode", "tfidf"))
    max_candidates_per_score = int(rep_cfg.get("max_candidates_per_score", 8))
    token_penalty = float(rep_cfg.get("token_cost_penalty", 0.02))
    score_values = sorted({int(x["domain1_score"]) for x in train})
    candidate_items = []
    for score in score_values:
        pool = [x for x in train if int(x["domain1_score"]) == score]
        pool = sorted(pool, key=lambda x: (token_len(x["essay_text"]), int(x["essay_id"])))
        candidate_items.extend(pool[:max_candidates_per_score])
    if not candidate_items:
        return [], []

    val_for_representation = list(val)
    if mode == "local_hidden" and backend is not None:
        val_sample = list(val[: int(rep_cfg.get("max_val_representations", 32))])
        val_for_representation = val_sample
        val_embs = []
        for item in val_sample:
            val_embs.append(
                backend.encode_scoring_context(
                    instruction=instruction,
                    static_exemplars=[],
                    essay_text=item["essay_text"],
                    score_min=score_min,
                    score_max=score_max,
                    representation_target="Encode the validation essay for anchor coverage.",
                ).numpy()
            )
        cand_embs = []
        for item in candidate_items:
            cand_embs.append(
                backend.encode_scoring_context(
                    instruction=instruction,
                    static_exemplars=[],
                    essay_text=item["essay_text"],
                    score_min=score_min,
                    score_max=score_max,
                    known_score=int(item["domain1_score"]),
                    representation_target="Encode this candidate reference essay for anchor selection.",
                ).numpy()
            )
        val_mat = np.vstack(val_embs) if val_embs else np.zeros((1, len(cand_embs[0])))
        cand_mat = np.vstack(cand_embs)
    else:
        all_items = list(candidate_items) + list(val)
        mat, _ = tfidf_vectors(all_items)
        cand_mat = mat[: len(candidate_items)]
        val_mat = mat[len(candidate_items):]

    high_threshold = _adaptive_high_score_threshold(
        {"data": {"score_min": score_min, "score_max": score_max}},
        [x["domain1_score"] for x in val],
    )
    selected: List[int] = []
    trace: List[Dict[str, Any]] = []
    selected_scores: set[int] = set()
    selected_bands: set[str] = set()
    centroid = val_mat.mean(axis=0)
    high_val = [
        i
        for i, item in enumerate(val_for_representation)
        if int(item["domain1_score"]) >= high_threshold
    ]
    high_centroid = val_mat[high_val].mean(axis=0) if high_val else centroid

    for step in range(min(k, len(candidate_items))):
        best_idx = None
        best_score = -1e9
        best_parts = {}
        for idx, item in enumerate(candidate_items):
            if idx in selected:
                continue
            score = int(item["domain1_score"])
            band = band_for(score, score_min, score_max)
            vec = cand_mat[idx]
            val_cov = float(np.dot(vec, centroid))
            high_cov = float(np.dot(vec, high_centroid)) if score >= high_threshold else 0.0
            score_cov = 0.35 if score not in selected_scores else 0.0
            band_cov = 0.20 if band not in selected_bands else 0.0
            diversity = 0.0
            redundancy = 0.0
            if selected:
                sims = [float(np.dot(vec, cand_mat[j])) for j in selected]
                redundancy = max(sims)
                diversity = 1.0 - redundancy
            length_pen = token_penalty * (token_len(item["essay_text"]) / 400.0)
            total = score_cov + band_cov + val_cov + 0.5 * high_cov + 0.2 * diversity - 0.25 * redundancy - length_pen
            if total > best_score:
                best_score = total
                best_idx = idx
                best_parts = {
                    "score_coverage_score": score_cov,
                    "band_coverage_score": band_cov,
                    "validation_hidden_coverage_score": val_cov,
                    "high_tail_coverage_score": high_cov,
                    "diversity_score": diversity,
                    "redundancy_penalty": redundancy,
                    "token_cost_penalty": length_pen,
                }
        if best_idx is None:
            break
        selected.append(best_idx)
        item = candidate_items[best_idx]
        selected_scores.add(int(item["domain1_score"]))
        selected_bands.add(band_for(int(item["domain1_score"]), score_min, score_max))
        trace.append(
            {
                "step": step + 1,
                "essay_id": int(item["essay_id"]),
                "gold_score": int(item["domain1_score"]),
                "selection_score": best_score,
                "selection_parts": best_parts,
                "representation_mode": mode,
            }
        )

    anchors = []
    for idx in selected:
        item = candidate_items[idx]
        tr = next(x for x in trace if x["essay_id"] == int(item["essay_id"]))
        anchors.append(
            AnchorRecord(
                essay_id=int(item["essay_id"]),
                gold_score=int(item["domain1_score"]),
                prompt_id=0,
                token_length=token_len(item["essay_text"]),
                source_split="train",
                selection_score=float(tr["selection_score"]),
                selection_reason=f"representation_guided:{mode}",
                essay_text=item["essay_text"],
            )
        )
    return anchors, trace


def _representation_candidate_scores(
    train: Sequence[Dict],
    val: Sequence[Dict],
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
) -> Tuple[List[Dict], List[Dict[str, Any]], str, List[str]]:
    rep_cfg = config.get("anchor_budget", {}).get("representation", {})
    mode = str(rep_cfg.get("mode", "tfidf"))
    max_candidates_per_score = int(rep_cfg.get("max_candidates_per_score", 8))
    token_penalty = float(rep_cfg.get("token_cost_penalty", 0.02))
    score_values = sorted({int(x["domain1_score"]) for x in train})
    candidate_items = []
    for score in score_values:
        pool = [x for x in train if int(x["domain1_score"]) == score]
        pool = sorted(pool, key=lambda x: (token_len(x["essay_text"]), int(x["essay_id"])))
        candidate_items.extend(pool[:max_candidates_per_score])
    if not candidate_items:
        return [], [], mode, []

    val_for_representation = list(val)
    if mode == "local_hidden" and backend is not None:
        val_sample = list(val[: int(rep_cfg.get("max_val_representations", 32))])
        val_for_representation = val_sample
        val_embs = []
        for item in val_sample:
            val_embs.append(
                backend.encode_scoring_context(
                    instruction=instruction,
                    static_exemplars=[],
                    essay_text=item["essay_text"],
                    score_min=score_min,
                    score_max=score_max,
                    representation_target="Encode the validation essay for anchor coverage.",
                ).numpy()
            )
        cand_embs = []
        for item in candidate_items:
            cand_embs.append(
                backend.encode_scoring_context(
                    instruction=instruction,
                    static_exemplars=[],
                    essay_text=item["essay_text"],
                    score_min=score_min,
                    score_max=score_max,
                    known_score=int(item["domain1_score"]),
                    representation_target="Encode this candidate reference essay for anchor selection.",
                ).numpy()
            )
        val_mat = np.vstack(val_embs) if val_embs else np.zeros((1, len(cand_embs[0])))
        cand_mat = np.vstack(cand_embs)
    else:
        all_items = list(candidate_items) + list(val)
        mat, _ = tfidf_vectors(all_items)
        cand_mat = mat[: len(candidate_items)]
        val_mat = mat[len(candidate_items):]

    high_threshold = _adaptive_high_score_threshold(
        {"data": {"score_min": score_min, "score_max": score_max}},
        [x["domain1_score"] for x in val],
    )
    centroid = val_mat.mean(axis=0)
    high_val = [
        i
        for i, item in enumerate(val_for_representation)
        if int(item["domain1_score"]) >= high_threshold
    ]
    high_centroid = val_mat[high_val].mean(axis=0) if high_val else centroid
    scored = []
    for idx, item in enumerate(candidate_items):
        score = int(item["domain1_score"])
        band = band_for(score, score_min, score_max)
        vec = cand_mat[idx]
        val_cov = float(np.dot(vec, centroid))
        high_cov = float(np.dot(vec, high_centroid)) if score >= high_threshold else 0.0
        length_pen = token_penalty * (token_len(item["essay_text"]) / 400.0)
        total = val_cov + 0.5 * high_cov - length_pen
        scored.append(
            {
                "candidate_index": idx,
                "item": item,
                "essay_id": int(item["essay_id"]),
                "gold_score": score,
                "band": band,
                "selection_score": float(total),
                "selection_parts": {
                    "validation_hidden_coverage_score": val_cov,
                    "high_tail_coverage_score": high_cov,
                    "token_cost_penalty": length_pen,
                },
                "representation_mode": mode,
            }
        )
    scored.sort(key=lambda x: (-float(x["selection_score"]), int(x["gold_score"]), int(x["essay_id"])))
    return candidate_items, scored, mode, ["validation_representation_coverage", "high_tail_coverage", mode]


def _band_quota(k: int, available_bands: Sequence[str]) -> Dict[str, int]:
    bands = [b for b in ["low", "mid", "high"] if b in set(available_bands)]
    if not bands or k <= 0:
        return {}
    base = k // len(bands)
    rem = k % len(bands)
    quotas = {band: base for band in bands}
    for band in bands[:rem]:
        quotas[band] += 1
    return quotas


def _minmax_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [1.0 for _ in values]
    return [(float(v) - lo) / (hi - lo) for v in values]


def _lexical_retrieval_rows(train: Sequence[Dict], val: Sequence[Dict]) -> List[Dict[str, Any]]:
    val_terms = Counter()
    for item in val:
        val_terms.update(tokenize_simple(item["essay_text"]))
    rows = []
    for item in train:
        terms = tokenize_simple(item["essay_text"])
        overlap = sum(val_terms[t] for t in terms)
        length_pen = 0.0005 * token_len(item["essay_text"])
        rows.append(
            {
                "item": item,
                "essay_id": int(item["essay_id"]),
                "gold_score": int(item["domain1_score"]),
                "retrieval_score": float(overlap - length_pen),
            }
        )
    rows.sort(key=lambda x: (-float(x["retrieval_score"]), int(x["gold_score"]), int(x["essay_id"])))
    return rows


def _text_jaccard(a: str, b: str) -> float:
    toks_a = tokenize_simple(a)
    toks_b = tokenize_simple(b)
    if not toks_a and not toks_b:
        return 0.0
    return len(toks_a & toks_b) / max(1, len(toks_a | toks_b))


def retrieval_grounded_stratified_rep_anchors(
    train: Sequence[Dict],
    val: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
    variant: str = "v2",
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]]]:
    if variant == "no_rep":
        rg_cfg = config.get("anchor_budget", {}).get(
            "retrieval_grounded_no_rep",
            config.get("anchor_budget", {}).get(
                "retrieval_grounded_rep_v21",
                config.get("anchor_budget", {}).get("retrieval_grounded_rep", {}),
            ),
        )
        default_retrieval_weight = 0.8
        default_representation_weight = 0.0
        default_fallback_margin = 0.05
        default_max_rep_replacements = 0
        reason_prefix = "retrieval_grounded_no_rep"
        method_has_representation = False
    elif variant == "v21":
        rg_cfg = config.get("anchor_budget", {}).get(
            "retrieval_grounded_rep_v21",
            config.get("anchor_budget", {}).get("retrieval_grounded_rep", {}),
        )
        default_retrieval_weight = 0.8
        default_representation_weight = 0.2
        default_fallback_margin = 0.05
        default_max_rep_replacements = min(3, max(0, k))
        reason_prefix = "retrieval_grounded_rep_v21"
        method_has_representation = True
    else:
        rg_cfg = config.get("anchor_budget", {}).get("retrieval_grounded_rep", {})
        default_retrieval_weight = 0.6
        default_representation_weight = 0.4
        default_fallback_margin = 0.03
        default_max_rep_replacements = max(0, k)
        reason_prefix = "retrieval_grounded_rep"
        method_has_representation = True
    rep_cfg = config.get("anchor_budget", {}).get("representation", {})
    mode = str(rep_cfg.get("mode", "tfidf"))
    per_slot_top_n = max(1, int(rg_cfg.get("per_slot_top_n", rg_cfg.get("per_band_top_n", 5))))
    retrieval_weight = float(rg_cfg.get("retrieval_weight", default_retrieval_weight))
    representation_weight = float(rg_cfg.get("representation_weight", default_representation_weight))
    coverage_bonus = float(rg_cfg.get("coverage_bonus", 0.1))
    redundancy_weight = float(rg_cfg.get("redundancy_weight", 0.2))
    token_weight = float(rg_cfg.get("token_weight", rep_cfg.get("token_cost_penalty", 0.02)))
    fallback_margin = float(rg_cfg.get("fallback_margin", rg_cfg.get("fallback_epsilon", default_fallback_margin)))
    max_rep_replacements = max(0, int(rg_cfg.get("max_rep_replacements", default_max_rep_replacements)))

    if k <= 0 or not train:
        return [], []

    retrieval_rows = _lexical_retrieval_rows(train, val)
    by_slot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for global_rank, row in enumerate(retrieval_rows, start=1):
        band = band_for(int(row["gold_score"]), score_min, score_max)
        slot = score_slot_for(int(row["gold_score"]), score_min, score_max)
        row = dict(row)
        row["band"] = band
        row["score_slot"] = slot
        row["retrieval_global_rank"] = global_rank
        by_slot[slot].append(row)

    retrieval_candidates: List[Dict[str, Any]] = []
    for slot in ["low_tail", "lower_mid", "median", "upper_mid", "high_tail", "max_or_top"]:
        for rank, row in enumerate(by_slot.get(slot, [])[:per_slot_top_n], start=1):
            row = dict(row)
            row["retrieval_rank"] = rank
            retrieval_candidates.append(row)
    if not retrieval_candidates:
        return [], []

    candidate_items = [row["item"] for row in retrieval_candidates]
    if not method_has_representation:
        rep_scores = [0.0 for _ in retrieval_candidates]
    elif mode == "local_hidden" and backend is not None:
        val_sample = list(val[: int(rep_cfg.get("max_val_representations", 32))])
        val_embs = [
            backend.encode_scoring_context(
                instruction=instruction,
                static_exemplars=[],
                essay_text=item["essay_text"],
                score_min=score_min,
                score_max=score_max,
                representation_target="Encode the validation essay for anchor coverage.",
            ).numpy()
            for item in val_sample
        ]
        cand_embs = [
            backend.encode_scoring_context(
                instruction=instruction,
                static_exemplars=[],
                essay_text=item["essay_text"],
                score_min=score_min,
                score_max=score_max,
                known_score=int(item["domain1_score"]),
                representation_target="Encode this retrieval candidate reference essay for anchor selection.",
            ).numpy()
            for item in candidate_items
        ]
        cand_mat = np.vstack(cand_embs)
        val_mat = np.vstack(val_embs) if val_embs else np.zeros((1, cand_mat.shape[1]))
    else:
        all_items = list(candidate_items) + list(val)
        mat, _ = tfidf_vectors(all_items)
        cand_mat = mat[: len(candidate_items)]
        val_mat = mat[len(candidate_items):]

    if method_has_representation:
        centroid = val_mat.mean(axis=0)
        rep_scores = [float(np.dot(cand_mat[i], centroid)) for i in range(len(retrieval_candidates))]
    for idx, row in enumerate(retrieval_candidates):
        row["candidate_index"] = idx
        row["representation_score"] = rep_scores[idx]
        row["retrieval_candidate_ids_in_slot"] = [
            int(x["essay_id"]) for x in by_slot.get(str(row["score_slot"]), [])[:per_slot_top_n]
        ]
        row["retrieval_candidate_ids_in_band"] = row["retrieval_candidate_ids_in_slot"]

    for slot in ["low_tail", "lower_mid", "median", "upper_mid", "high_tail", "max_or_top"]:
        rows = [row for row in retrieval_candidates if row["score_slot"] == slot]
        retrieval_norms = _minmax_normalize([float(row["retrieval_score"]) for row in rows])
        rep_norms = (
            _minmax_normalize([float(row["representation_score"]) for row in rows])
            if method_has_representation
            else [0.0 for _ in rows]
        )
        for row, retrieval_norm, rep_norm in zip(rows, retrieval_norms, rep_norms):
            row["normalized_retrieval_score"] = float(retrieval_norm)
            row["normalized_representation_score"] = float(rep_norm)

    quotas = score_slot_quota(k, [slot for slot, rows in by_slot.items() if rows])
    selected: List[Dict[str, Any]] = []
    used_ids: set[int] = set()
    trace: List[Dict[str, Any]] = []
    rep_replacements_used = 0

    for slot in ["low_tail", "lower_mid", "median", "upper_mid", "high_tail", "max_or_top"]:
        quota = quotas.get(slot, 0)
        for _ in range(quota):
            pool = [
                row for row in retrieval_candidates
                if row["score_slot"] == slot and int(row["essay_id"]) not in used_ids
            ]
            if not pool:
                break

            top_retrieval = sorted(
                pool,
                key=lambda x: (int(x["retrieval_rank"]), -float(x["retrieval_score"]), int(x["essay_id"])),
            )[0]
            scored_pool = []
            for row in pool:
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        _text_jaccard(str(row["item"]["essay_text"]), str(prev["item"]["essay_text"]))
                        for prev in selected
                    )
                token_penalty = token_weight * (token_len(row["item"]["essay_text"]) / 400.0)
                band_coverage_bonus = coverage_bonus if not any(x["score_slot"] == slot for x in selected) else 0.0
                combined = (
                    retrieval_weight * float(row.get("normalized_retrieval_score", 0.0))
                    + representation_weight * float(row.get("normalized_representation_score", 0.0))
                    + band_coverage_bonus
                    - redundancy_weight * redundancy
                    - token_penalty
                )
                scored = dict(row)
                scored["redundancy_score"] = float(redundancy)
                scored["token_cost_penalty"] = float(token_penalty)
                scored["coverage_bonus"] = float(band_coverage_bonus)
                scored["combined_score"] = float(combined)
                scored_pool.append(scored)

            best_combined = sorted(
                scored_pool,
                key=lambda x: (-float(x["combined_score"]), int(x["retrieval_rank"]), int(x["essay_id"])),
            )[0]
            top_retrieval_scored = next(x for x in scored_pool if int(x["essay_id"]) == int(top_retrieval["essay_id"]))
            use_retrieval_fallback = False
            fallback_reason = ""
            margin_to_retrieval_top = float(best_combined["combined_score"]) - float(top_retrieval_scored["combined_score"])
            selected_by_rep = method_has_representation and int(best_combined["essay_id"]) != int(top_retrieval_scored["essay_id"])
            replacement_index = 0
            selected_reason = f"{reason_prefix}: retrieval-first selection"
            if int(best_combined["essay_id"]) != int(top_retrieval_scored["essay_id"]):
                if margin_to_retrieval_top <= fallback_margin:
                    best_combined = top_retrieval_scored
                    use_retrieval_fallback = True
                    selected_by_rep = False
                    fallback_reason = "small_rerank_margin" if method_has_representation else "small_combined_margin_no_rep"
                    replacement_index = 0
                    selected_reason = f"{reason_prefix}: fallback to retrieval top because rerank margin was small"
                elif method_has_representation and rep_replacements_used >= max_rep_replacements:
                    best_combined = top_retrieval_scored
                    use_retrieval_fallback = True
                    selected_by_rep = False
                    fallback_reason = "max_rep_replacements_reached"
                    replacement_index = 0
                    selected_reason = f"{reason_prefix}: fallback to retrieval top because replacement cap was reached"
                else:
                    rep_replacements_used += 1
                    replacement_index = rep_replacements_used
                    selected_reason = f"{reason_prefix}: representation rerank within retrieval top-n"

            selected.append(best_combined)
            used_ids.add(int(best_combined["essay_id"]))
            trace.append(
                {
                    "step": len(trace) + 1,
                    "anchor_id": int(best_combined["essay_id"]),
                    "essay_id": int(best_combined["essay_id"]),
                    "score": int(best_combined["gold_score"]),
                    "gold_score": int(best_combined["gold_score"]),
                    "band": best_combined["band"],
                    "score_band": best_combined["band"],
                    "score_slot": best_combined["score_slot"],
                    "requested_band": best_combined["band"],
                    "requested_slot": slot,
                    "band_quota": quota,
                    "slot_quota": quota,
                    "retrieval_rank": int(best_combined["retrieval_rank"]),
                    "retrieval_score": float(best_combined["retrieval_score"]),
                    "representation_score": float(best_combined["representation_score"]),
                    "combined_score": float(best_combined["combined_score"]),
                    "redundancy_score": float(best_combined["redundancy_score"]),
                    "token_length": token_len(best_combined["item"]["essay_text"]),
                    "selected_reason": selected_reason,
                    "selected_by_rep_rerank": selected_by_rep,
                    "whether_selected_by_rep_rerank": selected_by_rep,
                    "fallback_to_retrieval": use_retrieval_fallback,
                    "whether_fallback_to_retrieval": use_retrieval_fallback,
                    "fallback_reason": fallback_reason,
                    "replacement_index": replacement_index,
                    "num_rep_replacements": rep_replacements_used,
                    "max_rep_replacements": max_rep_replacements,
                    "margin_to_retrieval_top": margin_to_retrieval_top,
                    "fallback_margin": fallback_margin,
                    "retrieval_weight": retrieval_weight,
                    "representation_weight": representation_weight,
                    "method_has_representation": method_has_representation,
                    "retrieval_candidate_ids_in_band": best_combined["retrieval_candidate_ids_in_band"],
                    "retrieval_candidate_ids_in_slot": best_combined["retrieval_candidate_ids_in_slot"],
                    "selection_parts": {
                        "normalized_retrieval_score": float(best_combined.get("normalized_retrieval_score", 0.0)),
                        "normalized_representation_score": float(best_combined.get("normalized_representation_score", 0.0)),
                        "coverage_bonus": float(best_combined["coverage_bonus"]),
                        "redundancy_penalty": float(best_combined["redundancy_score"]),
                        "token_cost_penalty": float(best_combined["token_cost_penalty"]),
                    },
                    "representation_mode": mode,
                }
            )
            if len(selected) >= k:
                break
        if len(selected) >= k:
            break

    if len(selected) < min(k, len(retrieval_candidates)):
        for row in retrieval_candidates:
            if int(row["essay_id"]) in used_ids:
                continue
            selected.append(row)
            used_ids.add(int(row["essay_id"]))
            trace.append(
                {
                    "step": len(trace) + 1,
                    "anchor_id": int(row["essay_id"]),
                    "essay_id": int(row["essay_id"]),
                    "score": int(row["gold_score"]),
                    "gold_score": int(row["gold_score"]),
                    "band": row["band"],
                    "score_band": row["band"],
                    "score_slot": row["score_slot"],
                    "requested_band": "fallback_any_band",
                    "requested_slot": "fallback_any_slot",
                    "band_quota": quotas,
                    "slot_quota": quotas,
                    "retrieval_rank": int(row["retrieval_rank"]),
                    "retrieval_score": float(row["retrieval_score"]),
                    "representation_score": float(row["representation_score"]),
                    "combined_score": float(row.get("combined_score", row["retrieval_score"])),
                    "redundancy_score": 0.0,
                    "token_length": token_len(row["item"]["essay_text"]),
                    "selected_reason": f"{reason_prefix}: fallback fill after quota underflow",
                    "selected_by_rep_rerank": False,
                    "whether_selected_by_rep_rerank": False,
                    "fallback_to_retrieval": True,
                    "whether_fallback_to_retrieval": True,
                    "replacement_index": 0,
                    "num_rep_replacements": rep_replacements_used,
                    "max_rep_replacements": max_rep_replacements,
                    "margin_to_retrieval_top": 0.0,
                    "fallback_margin": fallback_margin,
                    "retrieval_weight": retrieval_weight,
                    "representation_weight": representation_weight,
                    "method_has_representation": method_has_representation,
                    "retrieval_candidate_ids_in_band": row["retrieval_candidate_ids_in_band"],
                    "retrieval_candidate_ids_in_slot": row["retrieval_candidate_ids_in_slot"],
                    "selection_parts": {},
                    "representation_mode": mode,
                    "fallback_reason": "quota underfilled because one or more score bands lacked retrieval candidates",
                }
            )
            if len(selected) >= k:
                break

    anchors = [
        AnchorRecord(
            essay_id=int(row["item"]["essay_id"]),
            gold_score=int(row["item"]["domain1_score"]),
            prompt_id=0,
            token_length=token_len(row["item"]["essay_text"]),
            source_split="train",
            selection_score=float(row.get("combined_score", row.get("retrieval_score", 0.0))),
            selection_reason=f"{reason_prefix}_stratified:{mode}",
            essay_text=row["item"]["essay_text"],
        )
        for row in selected[:k]
    ]
    return anchors, trace


def retrieval_grounded_stratified_rep_anchors_v21(
    train: Sequence[Dict],
    val: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]]]:
    return retrieval_grounded_stratified_rep_anchors(
        train, val, k, score_min, score_max, config, instruction, backend, out_dir, variant="v21"
    )


def retrieval_grounded_no_rep_anchors(
    train: Sequence[Dict],
    val: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]]]:
    return retrieval_grounded_stratified_rep_anchors(
        train, val, k, score_min, score_max, config, instruction, backend, out_dir, variant="no_rep"
    )


def stability_retrieval_artifacts(
    train: Sequence[Dict],
    val: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    del instruction, backend
    cfg = config.get("anchor_budget", {}).get("stability_retrieval", {})
    stability_rows, stability_trace = estimate_anchor_stability(
        train,
        val,
        k=int(k),
        score_min=score_min,
        score_max=score_max,
        n_bootstrap=int(cfg.get("n_bootstrap", 8)),
        sample_ratio=float(cfg.get("sample_ratio", 0.75)),
        seed=int(cfg.get("seed", config.get("debug", {}).get("seed", 42))),
        per_band_top_n=int(cfg.get("per_slot_top_n", cfg.get("per_band_top_n", 8))),
        rank_variance_weight=float(cfg.get("rank_variance_weight", 0.10)),
        redundancy_weight=float(cfg.get("redundancy_weight", 0.15)),
    )
    selected_rows, selection_trace = select_stable_anchor_rows(
        train,
        stability_rows,
        k=int(k),
        score_min=score_min,
        score_max=score_max,
        token_weight=float(cfg.get("token_weight", config.get("anchor_budget", {}).get("representation", {}).get("token_cost_penalty", 0.02))),
        redundancy_weight=float(cfg.get("selection_redundancy_weight", 0.20)),
        tail_coverage_enabled=bool(cfg.get("tail_coverage_enabled", False)),
        min_top_score_anchors=int(cfg.get("min_top_score_anchors", 0)),
        top_score_margin=int(cfg.get("top_score_margin", 0)),
    )
    by_id = {int(item["essay_id"]): item for item in train}
    anchors: List[AnchorRecord] = []
    for row in selected_rows:
        item = by_id[int(row["essay_id"])]
        anchors.append(
            AnchorRecord(
                essay_id=int(item["essay_id"]),
                gold_score=int(item["domain1_score"]),
                prompt_id=0,
                token_length=token_len(item["essay_text"]),
                source_split="train",
                selection_score=float(row["stability_score"]),
                selection_reason="stability_retrieval:bootstrap",
                essay_text=item["essay_text"],
            )
        )
    return anchors, selection_trace, stability_rows, stability_trace


def stability_retrieval_anchors(
    train: Sequence[Dict],
    val: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]]]:
    anchors, selection_trace, stability_rows, stability_trace = stability_retrieval_artifacts(
        train, val, k, score_min, score_max, config, instruction, backend, out_dir
    )
    write_csv(out_dir / "anchor_stability_scores.csv", stability_rows)
    write_jsonl(out_dir / "stability_trace.jsonl", stability_trace)
    return anchors, selection_trace


def sisa_anchors(
    train: Sequence[Dict],
    val_diag: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]]]:
    cfg = config.get("anchor_budget", {}).get("sisa", {})
    stability_cfg = config.get("anchor_budget", {}).get("stability_retrieval", {})
    stability_rows, stability_trace = estimate_anchor_stability(
        train,
        val_diag,
        k=int(k),
        score_min=score_min,
        score_max=score_max,
        n_bootstrap=int(stability_cfg.get("n_bootstrap", 8)),
        sample_ratio=float(stability_cfg.get("sample_ratio", 0.75)),
        seed=int(stability_cfg.get("seed", config.get("debug", {}).get("seed", 42))),
        per_band_top_n=int(stability_cfg.get("per_slot_top_n", stability_cfg.get("per_band_top_n", 8))),
        rank_variance_weight=float(stability_cfg.get("rank_variance_weight", 0.10)),
        redundancy_weight=float(stability_cfg.get("redundancy_weight", 0.15)),
    )
    write_csv(out_dir / "anchor_stability_scores.csv", stability_rows)
    write_jsonl(out_dir / "stability_trace.jsonl", stability_trace)
    initial_rows, initial_trace = select_stable_anchor_rows(
        train,
        stability_rows,
        k=int(k),
        score_min=score_min,
        score_max=score_max,
        token_weight=float(stability_cfg.get("token_weight", config.get("anchor_budget", {}).get("representation", {}).get("token_cost_penalty", 0.02))),
        redundancy_weight=float(stability_cfg.get("selection_redundancy_weight", 0.20)),
        tail_coverage_enabled=bool(stability_cfg.get("tail_coverage_enabled", True)),
        min_top_score_anchors=int(stability_cfg.get("min_top_score_anchors", 1)),
        top_score_margin=int(stability_cfg.get("top_score_margin", 1)),
    )
    by_id = {int(item["essay_id"]): item for item in train}

    def rows_to_records(rows: Sequence[Dict[str, Any]], reason: str) -> List[AnchorRecord]:
        records = []
        for row in rows:
            item = by_id[int(row["essay_id"])]
            records.append(
                AnchorRecord(
                    essay_id=int(item["essay_id"]),
                    gold_score=int(item["domain1_score"]),
                    prompt_id=0,
                    token_length=token_len(item["essay_text"]),
                    source_split="train",
                    selection_score=float(row.get("combined_score", row.get("stability_score", 0.0)) or 0.0),
                    selection_reason=reason,
                    essay_text=item["essay_text"],
                )
            )
        return records

    initial_records = rows_to_records(initial_rows, "sisa:initial_stability_parent")
    initial_anchor_dicts = [anchor_record_to_bapr(record) for record in initial_records]
    write_json(
        out_dir / "sisa_initial_anchor_bank.json",
        {
            "method": "sisa_initial_stability_parent",
            "anchor_ids": [record.essay_id for record in initial_records],
            "anchor_scores": [record.gold_score for record in initial_records],
            "selection_trace_path": str(out_dir / "sisa_initial_selection_trace.jsonl"),
        },
    )
    write_jsonl(out_dir / "sisa_initial_selection_trace.jsonl", initial_trace)

    loo_rows: List[Dict[str, Any]] = []
    if bool(cfg.get("loo_parent_selection_enabled", True)) and initial_records:
        diag_rows, _ = score_items(backend, val_diag, instruction, initial_records, score_min, score_max)
        loo_rows, _ = compute_bapr_si_loo_attribution(
            backend=backend,
            val_diag=val_diag,
            diag_rows=diag_rows,
            instruction=instruction,
            parent_records=initial_records,
            parent_anchors=initial_anchor_dicts,
            proxy_influence_rows=[],
            score_min=score_min,
            score_max=score_max,
            config={
                **config,
                "bapr": {
                    **config.get("bapr", {}),
                    "influence": {
                        **config.get("bapr", {}).get("influence", {}),
                        "loo_attribution_enabled": True,
                        "loo_max_anchors": int(cfg.get("loo_max_anchors", len(initial_records))),
                        "loo_max_items": int(cfg.get("loo_max_items", len(val_diag))),
                    },
                },
            },
        )
    write_csv(out_dir / "sisa_loo_influence_scores.csv", loo_rows)

    selected_rows, trace = select_sisa_anchor_rows(
        train,
        stability_rows,
        loo_rows,
        k=int(k),
        score_min=score_min,
        score_max=score_max,
        stability_weight=float(cfg.get("stability_weight", 0.45)),
        retrieval_weight=float(cfg.get("retrieval_weight", 0.25)),
        influence_weight=float(cfg.get("influence_weight", 0.20)),
        redundancy_weight=float(cfg.get("redundancy_weight", 0.20)),
        token_weight=float(cfg.get("token_weight", stability_cfg.get("token_weight", 0.02))),
        tail_coverage_enabled=bool(cfg.get("tail_coverage_enabled", True)),
        min_top_score_anchors=int(cfg.get("min_top_score_anchors", stability_cfg.get("min_top_score_anchors", 1))),
        top_score_margin=int(cfg.get("top_score_margin", stability_cfg.get("top_score_margin", 1))),
    )
    anchors = rows_to_records(selected_rows, "sisa:stability_influence_scale_aware")
    return anchors, trace


def coverage_first_sisa_anchors(
    train: Sequence[Dict],
    val_diag: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]]]:
    cfg = config.get("anchor_budget", {}).get("coverage_first_sisa", {})
    stability_cfg = config.get("anchor_budget", {}).get("stability_retrieval", {})
    stability_rows, stability_trace = estimate_anchor_stability(
        train,
        val_diag,
        k=int(k),
        score_min=score_min,
        score_max=score_max,
        n_bootstrap=int(stability_cfg.get("n_bootstrap", 8)),
        sample_ratio=float(stability_cfg.get("sample_ratio", 0.75)),
        seed=int(stability_cfg.get("seed", config.get("debug", {}).get("seed", 42))),
        per_band_top_n=int(stability_cfg.get("per_slot_top_n", stability_cfg.get("per_band_top_n", 8))),
        rank_variance_weight=float(stability_cfg.get("rank_variance_weight", 0.10)),
        redundancy_weight=float(stability_cfg.get("redundancy_weight", 0.15)),
    )
    write_csv(out_dir / "anchor_stability_scores.csv", stability_rows)
    write_jsonl(out_dir / "stability_trace.jsonl", stability_trace)
    supported_scores = sorted(
        {
            int(item["domain1_score"])
            for item in list(train) + list(val_diag)
            if score_min <= int(item["domain1_score"]) <= score_max
        }
    )
    by_id = {int(item["essay_id"]): item for item in train}

    def rows_to_records(rows: Sequence[Dict[str, Any]], reason: str) -> List[AnchorRecord]:
        records: List[AnchorRecord] = []
        for row in rows:
            item = by_id[int(row["essay_id"])]
            records.append(
                AnchorRecord(
                    essay_id=int(item["essay_id"]),
                    gold_score=int(item["domain1_score"]),
                    prompt_id=0,
                    token_length=token_len(item["essay_text"]),
                    source_split="train",
                    selection_score=float(row.get("combined_score", row.get("stability_score", 0.0)) or 0.0),
                    selection_reason=reason,
                    essay_text=item["essay_text"],
                )
            )
        return records

    initial_rows, initial_trace = select_coverage_first_sisa_anchor_rows(
        train,
        stability_rows,
        [],
        k=int(k),
        score_min=score_min,
        score_max=score_max,
        supported_scores=supported_scores,
        ladder_strategy=str(cfg.get("target_ladder_strategy", "auto")),
        candidate_pool_per_score=int(cfg.get("candidate_pool_per_score", 8)),
        retrieval_weight=float(cfg.get("retrieval_weight", 0.45)),
        stability_weight=float(cfg.get("stability_weight", 0.25)),
        influence_weight=0.0,
        diversity_weight=float(cfg.get("diversity_weight", 0.08)),
        token_weight=float(cfg.get("token_weight", 0.02)),
    )
    initial_records = rows_to_records(initial_rows, "coverage_first_sisa:initial_no_loo_parent")
    write_json(
        out_dir / "coverage_first_sisa_initial_anchor_bank.json",
        {
            "method": "coverage_first_sisa_initial_no_loo_parent",
            "anchor_ids": [record.essay_id for record in initial_records],
            "anchor_scores": [record.gold_score for record in initial_records],
            "target_score_ladder": initial_trace[0].get("target_score_ladder", []) if initial_trace else [],
            "selection_trace_path": str(out_dir / "coverage_first_sisa_initial_selection_trace.jsonl"),
        },
    )
    write_jsonl(out_dir / "coverage_first_sisa_initial_selection_trace.jsonl", initial_trace)

    loo_rows: List[Dict[str, Any]] = []
    if bool(cfg.get("loo_enabled", True)) and initial_records:
        initial_anchor_dicts = [anchor_record_to_bapr(record) for record in initial_records]
        diag_rows, _ = score_items(backend, val_diag, instruction, initial_records, score_min, score_max)
        loo_rows, _ = compute_bapr_si_loo_attribution(
            backend=backend,
            val_diag=val_diag,
            diag_rows=diag_rows,
            instruction=instruction,
            parent_records=initial_records,
            parent_anchors=initial_anchor_dicts,
            proxy_influence_rows=[],
            score_min=score_min,
            score_max=score_max,
            config={
                **config,
                "bapr": {
                    **config.get("bapr", {}),
                    "influence": {
                        **config.get("bapr", {}).get("influence", {}),
                        "loo_attribution_enabled": True,
                        "loo_max_anchors": int(cfg.get("loo_max_anchors", len(initial_records))),
                        "loo_max_items": int(cfg.get("loo_max_items", len(val_diag))),
                    },
                },
            },
        )
    write_csv(out_dir / "coverage_first_sisa_loo_influence_scores.csv", loo_rows)

    selected_rows, trace = select_coverage_first_sisa_anchor_rows(
        train,
        stability_rows,
        loo_rows,
        k=int(k),
        score_min=score_min,
        score_max=score_max,
        supported_scores=supported_scores,
        ladder_strategy=str(cfg.get("target_ladder_strategy", "auto")),
        candidate_pool_per_score=int(cfg.get("candidate_pool_per_score", 8)),
        retrieval_weight=float(cfg.get("retrieval_weight", 0.45)),
        stability_weight=float(cfg.get("stability_weight", 0.25)),
        influence_weight=float(cfg.get("influence_weight", 0.20)),
        diversity_weight=float(cfg.get("diversity_weight", 0.08)),
        token_weight=float(cfg.get("token_weight", 0.02)),
    )
    anchors = rows_to_records(selected_rows, "coverage_first_sisa:scale_ladder_local_stability_influence")
    return anchors, trace


def stratified_rep_guided_anchors(
    train: Sequence[Dict],
    val: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]]]:
    _, scored, mode, _ = _representation_candidate_scores(
        train, val, score_min, score_max, config, instruction, backend
    )
    if not scored or k <= 0:
        return [], []
    by_band: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_band[str(row["band"])].append(row)
    quotas = _band_quota(k, [band for band, rows in by_band.items() if rows])
    selected: List[Dict[str, Any]] = []
    used_ids: set[int] = set()
    trace: List[Dict[str, Any]] = []

    for band in ["low", "mid", "high"]:
        quota = quotas.get(band, 0)
        for row in by_band.get(band, [])[:quota]:
            if int(row["essay_id"]) in used_ids:
                continue
            selected.append(row)
            used_ids.add(int(row["essay_id"]))
            trace.append(
                {
                    "step": len(trace) + 1,
                    "essay_id": int(row["essay_id"]),
                    "gold_score": int(row["gold_score"]),
                    "score_band": band,
                    "requested_band": band,
                    "band_quota": quota,
                    "fallback_reason": "",
                    "selection_score": float(row["selection_score"]),
                    "selection_parts": row["selection_parts"],
                    "representation_mode": mode,
                }
            )
            if len(selected) >= k:
                break
        if len(selected) >= k:
            break

    if len(selected) < k:
        for row in scored:
            if int(row["essay_id"]) in used_ids:
                continue
            selected.append(row)
            used_ids.add(int(row["essay_id"]))
            trace.append(
                {
                    "step": len(trace) + 1,
                    "essay_id": int(row["essay_id"]),
                    "gold_score": int(row["gold_score"]),
                    "score_band": row["band"],
                    "requested_band": "fallback_any_band",
                    "band_quota": quotas,
                    "fallback_reason": "quota underfilled because one or more score bands lacked available candidates",
                    "selection_score": float(row["selection_score"]),
                    "selection_parts": row["selection_parts"],
                    "representation_mode": mode,
                }
            )
            if len(selected) >= k:
                break

    anchors = []
    for row in selected[:k]:
        item = row["item"]
        anchors.append(
            AnchorRecord(
                essay_id=int(item["essay_id"]),
                gold_score=int(item["domain1_score"]),
                prompt_id=0,
                token_length=token_len(item["essay_text"]),
                source_split="train",
                selection_score=float(row["selection_score"]),
                selection_reason=f"stratified_rep_guided:{mode}",
                essay_text=item["essay_text"],
            )
        )
    return anchors, trace


def full_static_anchors(train: Sequence[Dict], score_min: int, score_max: int, per_score: int) -> List[AnchorRecord]:
    selected = []
    for score in range(score_min, score_max + 1):
        pool = sorted(
            [x for x in train if int(x["domain1_score"]) == score],
            key=lambda x: (token_len(x["essay_text"]), int(x["essay_id"])),
        )
        selected.extend(pool[:per_score])
    return [
        AnchorRecord(
            essay_id=int(x["essay_id"]),
            gold_score=int(x["domain1_score"]),
            prompt_id=0,
            token_length=token_len(x["essay_text"]),
            source_split="train",
            selection_score=1.0,
            selection_reason=f"full_static: up to {per_score} anchors per score level",
            essay_text=x["essay_text"],
        )
        for x in selected
    ]


def build_anchor_bank(
    method: str,
    k: Optional[int],
    train: Sequence[Dict],
    val: Sequence[Dict],
    score_min: int,
    score_max: int,
    seed: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]], bool, List[str]]:
    if method == "no_anchor":
        return [], [], False, []
    if method == "static_k_anchor":
        anchors = deterministic_score_covered(train, int(k or 0), score_min, score_max, seed, method)
        return anchors, [asdict(x) for x in anchors], False, ["score_coverage"]
    if method == "stratified_k_anchor":
        anchors = stratified_anchors(train, int(k or 0), score_min, score_max, seed)
        return anchors, [asdict(x) for x in anchors], False, ["score_band_coverage"]
    if method == "retrieval_k_anchor":
        anchors = retrieval_anchors(train, val, int(k or 0), score_min, score_max)
        return anchors, [asdict(x) for x in anchors], False, ["lexical_retrieval", "score_coverage"]
    if method == "representation_guided_k_anchor":
        anchors, trace = representation_guided_anchors(train, val, int(k or 0), score_min, score_max, config, instruction, backend, out_dir)
        static_ids = [x.essay_id for x in deterministic_score_covered(train, int(k or 0), score_min, score_max, seed, method)]
        rep_ids = [x.essay_id for x in anchors]
        mode = str(config.get("anchor_budget", {}).get("representation", {}).get("mode", "tfidf"))
        return anchors, trace, rep_ids != static_ids, ["score_coverage", "validation_representation_coverage", "high_tail_coverage", "diversity", mode]
    if method == "stratified_rep_guided_k_anchor":
        anchors, trace = stratified_rep_guided_anchors(train, val, int(k or 0), score_min, score_max, config, instruction, backend, out_dir)
        stratified_ids = [x.essay_id for x in stratified_anchors(train, int(k or 0), score_min, score_max, seed)]
        rep_ids = [x.essay_id for x in anchors]
        mode = str(config.get("anchor_budget", {}).get("representation", {}).get("mode", "tfidf"))
        return anchors, trace, rep_ids != stratified_ids, ["score_band_quota", "validation_representation_coverage", "high_tail_coverage", mode]
    if method == "retrieval_grounded_stratified_rep_k_anchor":
        anchors, trace = retrieval_grounded_stratified_rep_anchors(
            train, val, int(k or 0), score_min, score_max, config, instruction, backend, out_dir
        )
        retrieval_ids = [x.essay_id for x in retrieval_anchors(train, val, int(k or 0), score_min, score_max)]
        rep_ids = [x.essay_id for x in anchors]
        mode = str(config.get("anchor_budget", {}).get("representation", {}).get("mode", "tfidf"))
        return anchors, trace, rep_ids != retrieval_ids, [
            "score_band_quota",
            "retrieval_top_n",
            "representation_rerank",
            "redundancy_penalty",
            mode,
        ]
    if method == "retrieval_grounded_stratified_rep_k_anchor_v21":
        anchors, trace = retrieval_grounded_stratified_rep_anchors_v21(
            train, val, int(k or 0), score_min, score_max, config, instruction, backend, out_dir
        )
        retrieval_ids = [x.essay_id for x in retrieval_anchors(train, val, int(k or 0), score_min, score_max)]
        rep_ids = [x.essay_id for x in anchors]
        mode = str(config.get("anchor_budget", {}).get("representation", {}).get("mode", "tfidf"))
        return anchors, trace, rep_ids != retrieval_ids, [
            "score_band_quota",
            "retrieval_top_n",
            "conservative_representation_rerank",
            "replacement_cap",
            "redundancy_penalty",
            mode,
        ]
    if method == "retrieval_grounded_no_rep_k_anchor":
        anchors, trace = retrieval_grounded_no_rep_anchors(
            train, val, int(k or 0), score_min, score_max, config, instruction, backend, out_dir
        )
        retrieval_ids = [x.essay_id for x in retrieval_anchors(train, val, int(k or 0), score_min, score_max)]
        anchor_ids = [x.essay_id for x in anchors]
        return anchors, trace, anchor_ids != retrieval_ids, [
            "score_band_quota",
            "retrieval_top_n",
            "coverage_guard",
            "redundancy_penalty",
            "token_tiebreak",
            "no_representation",
        ]
    if method == "stability_retrieval_k_anchor":
        anchors, trace = stability_retrieval_anchors(
            train, val, int(k or 0), score_min, score_max, config, instruction, backend, out_dir
        )
        retrieval_ids = [x.essay_id for x in retrieval_anchors(train, val, int(k or 0), score_min, score_max)]
        stable_ids = [x.essay_id for x in anchors]
        return anchors, trace, stable_ids != retrieval_ids, [
            "retrieval_bootstrap_subsplits",
            "anchor_stability_estimator",
            "score_band_quota",
            "redundancy_penalty",
        ]
    if method == "sisa_k_anchor":
        anchors, trace = sisa_anchors(
            train, val, int(k or 0), score_min, score_max, config, instruction, backend, out_dir
        )
        retrieval_ids = [x.essay_id for x in retrieval_anchors(train, val, int(k or 0), score_min, score_max)]
        sisa_ids = [x.essay_id for x in anchors]
        return anchors, trace, sisa_ids != retrieval_ids, [
            "score_slot_ladder",
            "retrieval_relevance",
            "anchor_stability_estimator",
            "loo_influence_parent_selection",
            "diversity_penalty",
            "token_cost_penalty",
        ]
    if method == "coverage_first_sisa_k_anchor":
        anchors, trace = coverage_first_sisa_anchors(
            train, val, int(k or 0), score_min, score_max, config, instruction, backend, out_dir
        )
        retrieval_ids = [x.essay_id for x in retrieval_anchors(train, val, int(k or 0), score_min, score_max)]
        coverage_ids = [x.essay_id for x in anchors]
        return anchors, trace, coverage_ids != retrieval_ids, [
            "target_score_ladder",
            "exact_score_coverage_first",
            "local_retrieval_relevance",
            "anchor_stability_estimator",
            "loo_influence_local_tiebreak",
            "diversity_penalty",
            "token_cost_penalty",
        ]
    if method == "full_static_anchor":
        per_score = int(config.get("anchor_budget", {}).get("full_static_per_score", 3))
        anchors = full_static_anchors(train, score_min, score_max, per_score)
        return anchors, [asdict(x) for x in anchors], False, ["full_score_coverage"]
    raise ValueError(f"Unknown anchor method: {method}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_csv_with_fields(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_name_for(method: str, k: Optional[int]) -> str:
    return f"{method}_k{k}" if k is not None else method


REQUIRED_RUN_FILES = [
    "anchor_bank.json",
    "anchor_metrics.json",
    "anchor_selection_trace.jsonl",
    "predictions.csv",
    "prediction_distribution.csv",
    "score_boundary_metrics.json",
    "summary.md",
]


def is_run_complete(run_dir: Path, method: str) -> bool:
    required = list(REQUIRED_RUN_FILES)
    if method in {"bapr_repair_k_anchor", "bapr_si_k_anchor"}:
        required.extend(
            [
                "bapr_failure_profile.json",
                "bapr_repair_candidates.jsonl",
                "bapr_guarded_selection.csv",
                "bapr_parent_anchor_bank.json",
                "bapr_parent_metrics.json",
                "bapr_final_anchor_bank.json",
                "bapr_repair_trace.jsonl",
            ]
        )
    if method == "bapr_si_k_anchor":
        required.extend(
            [
                "anchor_stability_scores.csv",
                "stability_trace.jsonl",
                "anchor_influence_scores.csv",
                "anchor_influence_trace.jsonl",
                "anchor_loo_influence_scores.csv",
                "anchor_influence_child_alignment.csv",
                "bapr_si_repair_trace.jsonl",
            ]
        )
    if method in {
        "representation_guided_k_anchor",
        "stratified_rep_guided_k_anchor",
        "retrieval_grounded_stratified_rep_k_anchor",
        "retrieval_grounded_stratified_rep_k_anchor_v21",
        "retrieval_grounded_no_rep_k_anchor",
    }:
        required.extend(["representation_anchor_scores.csv", "representation_selection_trace.jsonl"])
    if method == "stability_retrieval_k_anchor":
        required.extend(["anchor_stability_scores.csv", "stability_trace.jsonl"])
    if method in {"sisa_k_anchor", "coverage_first_sisa_k_anchor"}:
        required.extend(
            [
                "anchor_stability_scores.csv",
                "stability_trace.jsonl",
            ]
        )
    if method == "sisa_k_anchor":
        required.extend(["sisa_initial_anchor_bank.json", "sisa_initial_selection_trace.jsonl", "sisa_loo_influence_scores.csv"])
    if method == "coverage_first_sisa_k_anchor":
        required.extend(
            [
                "coverage_first_sisa_initial_anchor_bank.json",
                "coverage_first_sisa_initial_selection_trace.jsonl",
                "coverage_first_sisa_loo_influence_scores.csv",
            ]
        )
    return all((run_dir / name).exists() for name in required)


def load_existing_run_summary(run_dir: Path, method: str, k: Optional[int]) -> Dict[str, Any]:
    metrics = read_json(run_dir / "score_boundary_metrics.json")
    anchor_bank = read_json(run_dir / "anchor_bank.json")
    anchor_metrics = read_json(run_dir / "anchor_metrics.json")
    predictions = []
    pred_path = run_dir / "predictions.csv"
    if pred_path.exists():
        with open(pred_path, encoding="utf-8", newline="") as f:
            predictions = list(csv.DictReader(f))
    total_tokens = sum(
        int(float(row.get("prompt_tokens", 0) or 0))
        + int(float(row.get("completion_tokens", 0) or 0))
        for row in predictions
    )
    val_metrics = metrics.get("val", {})
    test_metrics = metrics.get("test", {})
    return {
        "method": method,
        "k": int(k or anchor_metrics.get("anchor_count", anchor_bank.get("k", 0)) or 0),
        "exp_dir": str(run_dir),
        "val_qwk": float(val_metrics.get("qwk", 0.0) or 0.0),
        "test_qwk": float(test_metrics.get("qwk", 0.0) or 0.0),
        "mae": float(test_metrics.get("mae", 0.0) or 0.0),
        "high_recall": float(test_metrics.get("high_recall", 0.0) or 0.0),
        "max_recall": float(test_metrics.get("max_recall", 0.0) or 0.0),
        "SCI": float(test_metrics.get("score_compression_index", 0.0) or 0.0),
        "range_coverage": float(test_metrics.get("range_coverage", 0.0) or 0.0),
        "score_TV": float(test_metrics.get("score_tv", 0.0) or 0.0),
        "tokens": int(total_tokens),
        "runtime_sec": 0.0,
        "anchor_count": int(anchor_metrics.get("anchor_count", anchor_bank.get("k", 0)) or 0),
        "anchor_token_cost": int(anchor_bank.get("token_cost", 0) or 0),
        "representation_changed_anchor_choice": bool(
            anchor_bank.get("representation_changed_anchor_choice", False)
        ),
        "representation_features_used": anchor_bank.get("representation_features_used", []),
        "resumed_from_existing_outputs": True,
    }


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_items(
    backend: Optional[LocalLlamaBackend],
    items: Sequence[Dict],
    instruction: str,
    anchors: Sequence[AnchorRecord],
    score_min: int,
    score_max: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if backend is None:
        rows = []
        for item in items:
            gold = int(item["domain1_score"])
            # Deterministic fake scorer for smoke tests. It exercises IO and repair
            # control flow without loading or calling the LLM.
            pred = max(score_min, min(score_max, gold))
            rows.append(
                {
                    "essay_id": int(item["essay_id"]),
                    "gold_score": gold,
                    "pred_score": pred,
                    "raw_text": '{"final_score": %d, "fake_scoring": true}' % pred,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            )
        return rows, {"prompt_tokens": 0, "completion_tokens": 0, "representation_tokens": 0}
    before = backend.usage_snapshot()
    prompt_anchors = [a.to_prompt_example() for a in anchors]
    rows = []
    for item in items:
        result = backend.score(
            ScoringRequest(
                essay_id=int(item["essay_id"]),
                essay_text=item["essay_text"],
                instruction=instruction,
                static_exemplars=prompt_anchors,
                contrastive_anchors=[],
                score_min=score_min,
                score_max=score_max,
                dynamic_ex="(None)",
            )
        )
        rows.append(
            {
                "essay_id": int(item["essay_id"]),
                "gold_score": int(item["domain1_score"]),
                "pred_score": int(result.y_raw),
                "raw_text": result.raw_text,
                "prompt_tokens": int(result.meta.get("prompt_tokens", 0)),
                "completion_tokens": int(result.meta.get("completion_tokens", 0)),
            }
        )
    return rows, backend.usage_delta(before)


def anchor_record_to_bapr(anchor: AnchorRecord) -> Dict[str, Any]:
    return {
        "essay_id": int(anchor.essay_id),
        "gold_score": int(anchor.gold_score),
        "prompt_id": int(anchor.prompt_id),
        "token_length": int(anchor.token_length),
        "source_split": anchor.source_split,
        "selection_score": float(anchor.selection_score),
        "selection_reason": anchor.selection_reason,
        "essay_text": anchor.essay_text,
    }


def bapr_to_anchor_record(row: Dict[str, Any]) -> AnchorRecord:
    return AnchorRecord(
        essay_id=int(row["essay_id"]),
        gold_score=int(row["gold_score"]),
        prompt_id=int(row.get("prompt_id", 0)),
        token_length=int(row.get("token_length", token_len(row.get("essay_text", "")))),
        source_split=str(row.get("source_split", "train")),
        selection_score=float(row.get("selection_score", 0.0)),
        selection_reason=str(row.get("selection_reason", "bapr")),
        essay_text=str(row.get("essay_text", "")),
    )


def bapr_anchor_bank_payload(
    *,
    method: str,
    k: int,
    anchors: Sequence[Dict[str, Any]],
    score_min: int,
    score_max: int,
    prompt_id: int,
    split_hashes: Dict[str, Any],
    operator: str,
    parent_anchor_bank_id: str,
    trace_path: Path,
) -> Dict[str, Any]:
    anchor_ids = [int(a["essay_id"]) for a in anchors]
    slots = [score_slot_for(int(a["gold_score"]), score_min, score_max) for a in anchors]
    slot_labels = sorted(set(slots), key=lambda slot: ["low_tail", "lower_mid", "median", "upper_mid", "high_tail", "max_or_top"].index(slot))
    payload = {
        "anchor_bank_id": bapr_stable_hash(
            {
                "method": method,
                "prompt_id": prompt_id,
                "split_hashes": split_hashes,
                "k": k,
                "ordered_anchor_ids": anchor_ids,
                "operator": operator,
                "parent_anchor_bank_id": parent_anchor_bank_id,
            }
        ),
        "method": method,
        "k": int(k),
        "anchor_ids": anchor_ids,
        "anchor_scores": [int(a["gold_score"]) for a in anchors],
        "anchor_score_slots": slots,
        "anchor_slot_coverage": {slot: slots.count(slot) for slot in slot_labels},
        "score_coverage": score_coverage([bapr_to_anchor_record(a) for a in anchors], score_min, score_max),
        "token_cost": sum(int(a.get("token_length", token_len(a.get("essay_text", "")))) for a in anchors),
        "score_range": [score_min, score_max],
        "selection_trace_path": str(trace_path),
        "operator": operator,
        "parent_anchor_bank_id": parent_anchor_bank_id,
    }
    return payload


def build_bapr_parent_bank(
    *,
    parent_init_method: str,
    train: Sequence[Dict],
    val_diag: Sequence[Dict],
    k: int,
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    out_dir: Path,
) -> Tuple[List[AnchorRecord], List[Dict[str, Any]], str, str]:
    if parent_init_method in {"retrieval_grounded_stratified_rep_k_anchor_v21", "v21"}:
        records, trace = retrieval_grounded_stratified_rep_anchors_v21(
            train, val_diag, int(k), score_min, score_max, config, instruction, backend, out_dir
        )
        return records, trace, "BAPR-A0", "BAPR-A*"
    if parent_init_method == "retrieval_k_anchor":
        records = retrieval_anchors(train, val_diag, int(k), score_min, score_max)
        trace = [asdict(record) for record in records]
        return records, trace, "retrieval_diag_parent", "BAPR-retrieval-A*"
    if parent_init_method in {"stability_retrieval_k_anchor", "stable_retrieval"}:
        records, trace = stability_retrieval_anchors(
            train, val_diag, int(k), score_min, score_max, config, instruction, backend, out_dir
        )
        return records, trace, "BAPR-SI-A0", "BAPR-SI-A*"
    raise ValueError(f"Unsupported BAPR parent_init_method: {parent_init_method}")


def _merge_usage(*usages: Dict[str, int]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for usage in usages:
        for key, value in (usage or {}).items():
            merged[key] = int(merged.get(key, 0) + int(value or 0))
    return merged


def compute_bapr_si_loo_attribution(
    *,
    backend: Optional[LocalLlamaBackend],
    val_diag: Sequence[Dict],
    diag_rows: Sequence[Dict[str, Any]],
    instruction: str,
    parent_records: Sequence[AnchorRecord],
    parent_anchors: Sequence[Dict[str, Any]],
    proxy_influence_rows: Sequence[Dict[str, Any]],
    score_min: int,
    score_max: int,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    cfg = config.get("bapr", {}).get("influence", {})
    enabled = bool(cfg.get("loo_attribution_enabled", False))
    if not enabled or not parent_records:
        return [], {}

    max_items = int(cfg.get("loo_max_items", len(val_diag)))
    max_anchors = int(cfg.get("loo_max_anchors", len(parent_records)))
    loo_items = list(val_diag[: max(1, min(max_items, len(val_diag)))])
    loo_ids = {int(item["essay_id"]) for item in loo_items}
    diag_by_id = {int(row["essay_id"]): row for row in diag_rows}
    if len(loo_items) == len(val_diag) and all(int(item["essay_id"]) in diag_by_id for item in loo_items):
        parent_rows = [diag_by_id[int(item["essay_id"])] for item in loo_items]
        parent_usage: Dict[str, int] = {}
    else:
        parent_rows, parent_usage = score_items(backend, loo_items, instruction, parent_records, score_min, score_max)
    parent_metrics = metrics_with_anchor_stats(
        [row["gold_score"] for row in parent_rows],
        [row["pred_score"] for row in parent_rows],
        parent_anchors,
        score_min,
        score_max,
    )

    loo_metrics_by_anchor: Dict[int, Dict[str, Any]] = {}
    loo_usage: Dict[str, int] = dict(parent_usage)
    anchor_order = [int(anchor.essay_id) for anchor in parent_records[: max(0, max_anchors)]]
    for anchor_id in anchor_order:
        records_wo = [record for record in parent_records if int(record.essay_id) != anchor_id]
        anchors_wo = [anchor for anchor in parent_anchors if int(anchor["essay_id"]) != anchor_id]
        rows_wo, usage_wo = score_items(backend, loo_items, instruction, records_wo, score_min, score_max)
        loo_usage = _merge_usage(loo_usage, usage_wo)
        loo_metrics_by_anchor[anchor_id] = metrics_with_anchor_stats(
            [row["gold_score"] for row in rows_wo],
            [row["pred_score"] for row in rows_wo],
            anchors_wo,
            score_min,
            score_max,
        )

    proxy_by_id = {int(row["anchor_id"]): row for row in proxy_influence_rows if "anchor_id" in row}
    base_rows = estimate_leave_one_anchor_out_influence(parent_metrics, loo_metrics_by_anchor)
    output_rows: List[Dict[str, Any]] = []
    for row in base_rows:
        anchor_id = int(row["anchor_id"])
        proxy = proxy_by_id.get(anchor_id, {})
        metrics_wo = loo_metrics_by_anchor.get(anchor_id, {})
        parent_anchor = next((anchor for anchor in parent_anchors if int(anchor["essay_id"]) == anchor_id), {})
        delta_qwk = float(row.get("delta_qwk_without_anchor", 0.0) or 0.0)
        delta_mae = float(row.get("delta_mae_without_anchor", 0.0) or 0.0)
        delta_high = float(row.get("delta_high_recall_without_anchor", 0.0) or 0.0)
        delta_tv = float(row.get("delta_score_tv_without_anchor", 0.0) or 0.0)
        loo_harm_score = delta_qwk - delta_mae + delta_high - delta_tv
        output_rows.append(
            {
                **row,
                "anchor_score": parent_anchor.get("gold_score"),
                "anchor_band": band_for(int(parent_anchor.get("gold_score", score_min)), score_min, score_max)
                if parent_anchor
                else "",
                "anchor_slot": score_slot_for(int(parent_anchor.get("gold_score", score_min)), score_min, score_max)
                if parent_anchor
                else "",
                "loo_eval_n": len(loo_items),
                "loo_eval_ids_hash": id_list_hash(sorted(loo_ids)),
                "parent_qwk": parent_metrics.get("qwk"),
                "parent_mae": parent_metrics.get("mae"),
                "parent_high_recall": parent_metrics.get("high_recall"),
                "parent_score_tv": parent_metrics.get("score_tv"),
                "without_anchor_qwk": metrics_wo.get("qwk"),
                "without_anchor_mae": metrics_wo.get("mae"),
                "without_anchor_high_recall": metrics_wo.get("high_recall"),
                "without_anchor_score_tv": metrics_wo.get("score_tv"),
                "loo_harm_score": loo_harm_score,
                "proxy_failure_type": proxy.get("anchor_failure_type"),
                "proxy_negative_influence_score": proxy.get("negative_influence_score"),
                "proxy_stability_score": proxy.get("stability_score"),
            }
        )
    output_rows.sort(
        key=lambda row: (
            -float(row.get("loo_harm_score", 0.0) or 0.0),
            -float(row.get("proxy_negative_influence_score", 0.0) or 0.0),
            int(row["anchor_id"]),
        )
    )
    return output_rows, loo_usage


def build_influence_child_alignment_rows(
    *,
    children: Sequence[Dict[str, Any]],
    child_metrics_list: Sequence[Dict[str, Any]],
    parent_metrics: Dict[str, Any],
    guard_rows: Sequence[Dict[str, Any]],
    proxy_influence_rows: Sequence[Dict[str, Any]],
    loo_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    guard_by_id = {str(row.get("candidate_id")): row for row in guard_rows}
    proxy_by_id = {int(row["anchor_id"]): row for row in proxy_influence_rows if "anchor_id" in row}
    loo_by_id = {int(row["anchor_id"]): row for row in loo_rows if "anchor_id" in row}
    rows: List[Dict[str, Any]] = []
    for child, metrics in zip(children, child_metrics_list):
        child_id = str(child.get("candidate_id", ""))
        removed_id = child.get("removed_anchor_id")
        removed_id_int = int(removed_id) if removed_id not in (None, "") else None
        proxy = proxy_by_id.get(removed_id_int or -1, {})
        loo = loo_by_id.get(removed_id_int or -1, {})
        expected_metric = str(child.get("expected_repair_metric") or "")
        rows.append(
            {
                "candidate_id": child_id,
                "operator": child.get("operator"),
                "removed_anchor_id": removed_id_int,
                "added_anchor_id": child.get("added_anchor_id"),
                "target_slot": child.get("target_slot"),
                "proxy_failure_type": proxy.get("anchor_failure_type"),
                "proxy_negative_influence_score": proxy.get("negative_influence_score"),
                "loo_harm_score": loo.get("loo_harm_score"),
                "loo_vetoed": child.get("loo_vetoed", proxy.get("loo_vetoed")),
                "proxy_disagrees_with_loo": child.get("proxy_disagrees_with_loo", proxy.get("proxy_disagrees_with_loo")),
                "delta_qwk_without_anchor": loo.get("delta_qwk_without_anchor"),
                "child_delta_qwk": float(metrics.get("qwk", 0.0) or 0.0) - float(parent_metrics.get("qwk", 0.0) or 0.0),
                "child_delta_mae": float(metrics.get("mae", 0.0) or 0.0) - float(parent_metrics.get("mae", 0.0) or 0.0),
                "child_delta_high_recall": float(metrics.get("high_recall", 0.0) or 0.0)
                - float(parent_metrics.get("high_recall", 0.0) or 0.0),
                "expected_repair_metric": expected_metric,
                "child_delta_expected_metric": (
                    float(metrics.get(expected_metric, 0.0) or 0.0)
                    - float(parent_metrics.get(expected_metric, 0.0) or 0.0)
                    if expected_metric
                    else None
                ),
                "accepted_by_guard": guard_by_id.get(child_id, {}).get("accepted_by_guard"),
                "target_boundary_metric_improved": guard_by_id.get(child_id, {}).get("target_boundary_metric_improved"),
                "guard_reject_reasons": guard_by_id.get(child_id, {}).get("guard_reject_reasons"),
            }
        )
    return rows


def run_bapr_one(
    *,
    method: str,
    k: int,
    config: Dict[str, Any],
    fold: int,
    seed: int,
    train: Sequence[Dict],
    val: Sequence[Dict],
    test: Sequence[Dict],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    root_out: Path,
    split_hashes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    score_min = int(config["data"]["score_min"])
    score_max = int(config["data"]["score_max"])
    prompt_id = int(config["data"]["essay_set"])
    run_name = run_name_for(method, k)
    out_dir = root_out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    val_diag, val_sel, split_meta = split_val_diag_sel(
        val,
        score_min,
        score_max,
        float(config.get("bapr", {}).get("val_diag_ratio", 0.5)),
    )
    write_json(out_dir / "bapr_val_split.json", split_meta)
    forbidden_ids = {int(x["essay_id"]) for x in val_diag} | {int(x["essay_id"]) for x in val_sel} | {int(x["essay_id"]) for x in test}

    is_bapr_si = method == "bapr_si_k_anchor" or str(config.get("bapr", {}).get("repair_mode", "")) == "stability_influence"
    default_parent_method = "stability_retrieval_k_anchor" if is_bapr_si else "retrieval_grounded_stratified_rep_k_anchor_v21"
    parent_init_method = str(config.get("bapr", {}).get("parent_init_method", default_parent_method))
    stability_rows: List[Dict[str, Any]] = []
    stability_trace: List[Dict[str, Any]] = []
    if is_bapr_si and parent_init_method in {"stability_retrieval_k_anchor", "stable_retrieval"}:
        parent_records, parent_trace, stability_rows, stability_trace = stability_retrieval_artifacts(
            train,
            val_diag,
            int(k),
            score_min,
            score_max,
            config,
            instruction,
            backend,
            out_dir,
        )
        write_csv(out_dir / "anchor_stability_scores.csv", stability_rows)
        write_jsonl(out_dir / "stability_trace.jsonl", stability_trace)
        parent_candidate_id = "BAPR-SI-A0"
        final_method_name = "BAPR-SI-A*"
    else:
        parent_records, parent_trace, parent_candidate_id, final_method_name = build_bapr_parent_bank(
            parent_init_method=parent_init_method,
            train=train,
            val_diag=val_diag,
            k=int(k),
            score_min=score_min,
            score_max=score_max,
            config=config,
            instruction=instruction,
            backend=backend,
            out_dir=out_dir,
        )
    parent_anchors = [anchor_record_to_bapr(a) for a in parent_records]
    parent_trace_path = out_dir / "anchor_selection_trace.jsonl"
    write_jsonl(parent_trace_path, parent_trace)
    parent_bank = bapr_anchor_bank_payload(
        method=parent_candidate_id,
        k=int(k),
        anchors=parent_anchors,
        score_min=score_min,
        score_max=score_max,
        prompt_id=prompt_id,
        split_hashes={**(split_hashes or {}), **split_meta},
        operator="PARENT",
        parent_anchor_bank_id="",
        trace_path=parent_trace_path,
    )
    parent_bank["candidate_id"] = parent_candidate_id
    parent_bank["bapr_parent_init_method"] = parent_init_method
    write_json(out_dir / "bapr_parent_anchor_bank.json", parent_bank)

    diag_rows, diag_usage = score_items(backend, val_diag, instruction, parent_records, score_min, score_max)
    failure_profile = compute_failure_profile(
        [r["gold_score"] for r in diag_rows],
        [r["pred_score"] for r in diag_rows],
        parent_anchors,
        score_min,
        score_max,
        parent_trace,
    )
    write_json(out_dir / "bapr_failure_profile.json", failure_profile)
    ranked = rank_repair_operators(failure_profile)
    influence_rows: List[Dict[str, Any]] = []
    influence_trace: List[Dict[str, Any]] = []
    loo_influence_rows: List[Dict[str, Any]] = []
    loo_usage: Dict[str, int] = {}
    if is_bapr_si:
        if not stability_rows:
            stability_cfg = config.get("anchor_budget", {}).get("stability_retrieval", {})
            stability_rows, stability_trace = estimate_anchor_stability(
                train,
                val_diag,
                k=int(k),
                score_min=score_min,
                score_max=score_max,
                n_bootstrap=int(stability_cfg.get("n_bootstrap", 8)),
                sample_ratio=float(stability_cfg.get("sample_ratio", 0.75)),
                seed=int(stability_cfg.get("seed", config.get("debug", {}).get("seed", 42))),
                per_band_top_n=int(stability_cfg.get("per_slot_top_n", stability_cfg.get("per_band_top_n", 8))),
                rank_variance_weight=float(stability_cfg.get("rank_variance_weight", 0.10)),
                redundancy_weight=float(stability_cfg.get("redundancy_weight", 0.15)),
            )
            write_csv(out_dir / "anchor_stability_scores.csv", stability_rows)
            write_jsonl(out_dir / "stability_trace.jsonl", stability_trace)
        influence_rows, influence_trace = estimate_proxy_influence(
            parent_anchors,
            train,
            val_diag,
            failure_profile,
            stability_rows,
            score_min=score_min,
            score_max=score_max,
        )
        write_csv(out_dir / "anchor_influence_scores.csv", influence_rows)
        write_jsonl(out_dir / "anchor_influence_trace.jsonl", influence_trace)
        loo_influence_rows, loo_usage = compute_bapr_si_loo_attribution(
            backend=backend,
            val_diag=val_diag,
            diag_rows=diag_rows,
            instruction=instruction,
            parent_records=parent_records,
            parent_anchors=parent_anchors,
            proxy_influence_rows=influence_rows,
            score_min=score_min,
            score_max=score_max,
            config=config,
        )
        write_csv_with_fields(
            out_dir / "anchor_loo_influence_scores.csv",
            loo_influence_rows,
            [
                "anchor_id",
                "anchor_score",
                "anchor_band",
                "anchor_slot",
                "loo_eval_n",
                "loo_eval_ids_hash",
                "parent_qwk",
                "without_anchor_qwk",
                "delta_qwk_without_anchor",
                "parent_mae",
                "without_anchor_mae",
                "delta_mae_without_anchor",
                "parent_high_recall",
                "without_anchor_high_recall",
                "delta_high_recall_without_anchor",
                "parent_score_tv",
                "without_anchor_score_tv",
                "delta_score_tv_without_anchor",
                "loo_harm_score",
                "proxy_failure_type",
                "proxy_negative_influence_score",
                "proxy_stability_score",
            ],
        )
        influence_rows = apply_loo_veto_to_proxy_influence(
            influence_rows,
            loo_influence_rows,
            enabled=bool(config.get("bapr", {}).get("influence", {}).get("loo_veto_enabled", True)),
        )
        write_csv(out_dir / "anchor_influence_scores.csv", influence_rows)
        children, repair_trace = generate_influence_repair_children(
            parent_anchors,
            train,
            val_diag,
            stability_rows,
            influence_rows,
            score_min=score_min,
            score_max=score_max,
            k=int(k),
            forbidden_ids=forbidden_ids,
            max_children=int(config.get("bapr", {}).get("max_children", 3)),
        )
        write_jsonl(out_dir / "bapr_si_repair_trace.jsonl", repair_trace)
    else:
        children, repair_trace = generate_repaired_children(
            parent_anchors,
            train,
            val_diag,
            failure_profile,
            ranked,
            score_min,
            score_max,
            int(k),
            forbidden_ids=forbidden_ids,
            max_children=int(config.get("bapr", {}).get("max_children", 3)),
        )
    for child in children:
        child["parent_id"] = parent_candidate_id
        child["bapr_parent_init_method"] = parent_init_method
    write_jsonl(out_dir / "bapr_repair_trace.jsonl", repair_trace)
    write_jsonl(out_dir / "bapr_repair_candidates.jsonl", children)

    parent_sel_rows, parent_sel_usage = score_items(backend, val_sel, instruction, parent_records, score_min, score_max)
    parent_metrics = metrics_with_anchor_stats(
        [r["gold_score"] for r in parent_sel_rows],
        [r["pred_score"] for r in parent_sel_rows],
        parent_anchors,
        score_min,
        score_max,
    )
    stability_lookup = stability_by_id(stability_rows) if is_bapr_si else {}

    def add_anchor_stability_metrics(metrics: Dict[str, Any], anchors: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not is_bapr_si:
            return metrics
        values = [
            float(stability_lookup.get(int(anchor["essay_id"]), {}).get("stability_score", 0.0) or 0.0)
            for anchor in anchors
        ]
        freqs = [
            float(stability_lookup.get(int(anchor["essay_id"]), {}).get("selection_frequency", 0.0) or 0.0)
            for anchor in anchors
        ]
        metrics["anchor_stability_score"] = float(np.mean(values)) if values else 0.0
        metrics["anchor_stability_min"] = float(min(values)) if values else 0.0
        metrics["anchor_selection_frequency_mean"] = float(np.mean(freqs)) if freqs else 0.0
        return metrics

    parent_metrics = add_anchor_stability_metrics(parent_metrics, parent_anchors)
    write_json(out_dir / "bapr_parent_metrics.json", parent_metrics)
    child_metrics_list = []
    child_sel_rows_list = []
    for child in children:
        child_records = [bapr_to_anchor_record(a) for a in child["anchors"]]
        rows, _ = score_items(backend, val_sel, instruction, child_records, score_min, score_max)
        child_sel_rows_list.append(rows)
        metrics = metrics_with_anchor_stats(
            [r["gold_score"] for r in rows],
            [r["pred_score"] for r in rows],
            child["anchors"],
            score_min,
            score_max,
        )
        metrics = add_anchor_stability_metrics(metrics, child["anchors"])
        child_metrics_list.append(metrics)

    guard = guarded_select_repaired_bank(
        parent_metrics,
        child_metrics_list,
        parent_bank,
        children,
        config,
        parent_eval_rows=parent_sel_rows,
        child_eval_rows_list=child_sel_rows_list,
        score_min=score_min,
        score_max=score_max,
    )
    write_csv(out_dir / "bapr_guarded_selection.csv", guard["selection_rows"])
    if is_bapr_si:
        write_csv_with_fields(
            out_dir / "anchor_influence_child_alignment.csv",
            build_influence_child_alignment_rows(
                children=children,
                child_metrics_list=child_metrics_list,
                parent_metrics=parent_metrics,
                guard_rows=guard["selection_rows"],
                proxy_influence_rows=influence_rows,
                loo_rows=loo_influence_rows,
            ),
            [
                "candidate_id",
                "operator",
                "removed_anchor_id",
                "added_anchor_id",
                "target_slot",
                "proxy_failure_type",
                "proxy_negative_influence_score",
                "loo_harm_score",
                "loo_vetoed",
                "proxy_disagrees_with_loo",
                "delta_qwk_without_anchor",
                "child_delta_qwk",
                "child_delta_mae",
                "child_delta_high_recall",
                "expected_repair_metric",
                "child_delta_expected_metric",
                "accepted_by_guard",
                "target_boundary_metric_improved",
                "guard_reject_reasons",
            ],
        )
    selected_bank = guard["selected_anchor_bank"]
    final_anchors = parent_anchors if selected_bank.get("candidate_id") is None else selected_bank.get("anchors", parent_anchors)
    if selected_bank.get("anchor_ids") == parent_bank["anchor_ids"]:
        final_anchors = parent_anchors
    final_records = [bapr_to_anchor_record(a) for a in final_anchors]
    final_trace_path = out_dir / "bapr_repair_trace.jsonl"
    final_bank = bapr_anchor_bank_payload(
        method=final_method_name,
        k=int(k),
        anchors=final_anchors,
        score_min=score_min,
        score_max=score_max,
        prompt_id=prompt_id,
        split_hashes={**(split_hashes or {}), **split_meta},
        operator=str(selected_bank.get("operator", "PARENT")),
        parent_anchor_bank_id=parent_bank["anchor_bank_id"],
        trace_path=final_trace_path,
    )
    final_bank["selected_reason"] = guard["selected_reason"]
    final_bank["bapr_parent_init_method"] = parent_init_method
    final_bank["parent_candidate_id"] = parent_candidate_id
    if is_bapr_si:
        final_bank["bapr_repair_mode"] = "stability_influence"
        final_bank["anchor_stability_score"] = guard["selected_metrics"].get("anchor_stability_score")
        final_bank["anchor_stability_min"] = guard["selected_metrics"].get("anchor_stability_min")
        final_bank["anchor_selection_frequency_mean"] = guard["selected_metrics"].get("anchor_selection_frequency_mean")
    write_json(out_dir / "bapr_final_anchor_bank.json", final_bank)

    write_json(out_dir / "anchor_bank.json", final_bank)
    write_json(
        out_dir / "anchor_metrics.json",
        {
            **final_bank,
            "anchor_count": len(final_records),
            "average_anchor_length": float(np.mean([a.token_length for a in final_records])) if final_records else 0.0,
            "selected_anchors": [asdict(a) for a in final_records],
            "anchor_replacement_count": len(set(parent_bank["anchor_ids"]) - set(final_bank["anchor_ids"])),
        },
    )

    t0 = time.time()
    final_sel_rows, final_sel_usage = score_items(backend, val_sel, instruction, final_records, score_min, score_max)
    test_rows, test_usage = score_items(backend, test, instruction, final_records, score_min, score_max)
    runtime = time.time() - t0
    pred_rows = [{"split": "val_diag_parent", **row} for row in diag_rows]
    pred_rows.extend({"split": "val_sel_final", **row} for row in final_sel_rows)
    pred_rows.extend({"split": "test", **row} for row in test_rows)
    write_csv(out_dir / "predictions.csv", pred_rows)
    write_csv(out_dir / "test_predictions.csv", [row for row in pred_rows if row["split"] == "test"])

    val_metrics = guard["selected_metrics"]
    test_metrics = score_boundary_metrics(
        [r["gold_score"] for r in test_rows],
        [r["pred_score"] for r in test_rows],
        score_min,
        score_max,
    )
    write_json(out_dir / "score_boundary_metrics.json", {"val": val_metrics, "test": test_metrics})
    dist_rows = []
    for split, metrics in [("val_sel", val_metrics), ("test", test_metrics)]:
        for score in range(score_min, score_max + 1):
            dist_rows.append(
                {
                    "split": split,
                    "score": score,
                    "gold_count": metrics["gold_distribution"].get(str(score), 0) if "gold_distribution" in metrics else None,
                    "pred_count": metrics["prediction_distribution"].get(str(score), 0) if "prediction_distribution" in metrics else None,
                    "per_score_recall": None,
                }
            )
    write_csv(out_dir / "prediction_distribution.csv", dist_rows)
    usage = {
        key: int(
            diag_usage.get(key, 0)
            + loo_usage.get(key, 0)
            + parent_sel_usage.get(key, 0)
            + final_sel_usage.get(key, 0)
            + test_usage.get(key, 0)
        )
        for key in set(diag_usage) | set(loo_usage) | set(parent_sel_usage) | set(final_sel_usage) | set(test_usage)
    }
    total_tokens = int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0) + usage.get("representation_tokens", 0))
    summary = {
        "method": method,
        "k": int(k),
        "exp_dir": str(out_dir),
        "val_qwk": float(val_metrics.get("qwk", 0.0) or 0.0),
        "test_qwk": test_metrics["qwk"],
        "mae": test_metrics["mae"],
        "high_recall": test_metrics["high_recall"],
        "max_recall": test_metrics["max_recall"],
        "SCI": test_metrics["score_compression_index"],
        "range_coverage": test_metrics["range_coverage"],
        "score_TV": test_metrics["score_tv"],
        "tokens": total_tokens,
        "runtime_sec": runtime,
        "anchor_count": len(final_records),
        "anchor_token_cost": final_bank["token_cost"],
        "anchor_slot_coverage": val_metrics.get("anchor_slot_coverage"),
        "anchor_required_slot_count": val_metrics.get("anchor_required_slot_count"),
        "missing_anchor_slots": val_metrics.get("missing_anchor_slots"),
        "representation_changed_anchor_choice": final_bank["anchor_ids"] != parent_bank["anchor_ids"],
        "representation_features_used": [
            "bapr_repair",
            "score_boundary_diagnostics",
            "anchor_stability_estimator",
            "anchor_influence_estimator",
        ] if is_bapr_si else ["bapr_repair", "score_boundary_diagnostics"],
        "anchor_stability_score": final_bank.get("anchor_stability_score"),
        "bapr_selected_reason": guard["selected_reason"],
        "bapr_parent_init_method": parent_init_method,
        "bapr_parent_candidate_id": parent_candidate_id,
        "resumed_from_existing_outputs": False,
    }
    write_json(out_dir / "run_summary.json", summary)
    (out_dir / "summary.md").write_text(
        "# Summary\n\n"
        "## Goal\nBAPR one-step anchor-bank repair under fixed instruction and fixed raw LLM scoring.\n\n"
        f"## Results\n- selected reason: {guard['selected_reason']}\n"
        f"- val_sel QWK: {summary['val_qwk']:.4f}\n- raw test QWK: {summary['test_qwk']:.4f}\n",
        encoding="utf-8",
    )
    return summary


def run_one(
    *,
    method: str,
    k: Optional[int],
    config: Dict[str, Any],
    fold: int,
    seed: int,
    train: Sequence[Dict],
    val: Sequence[Dict],
    test: Sequence[Dict],
    instruction: str,
    backend: Optional[LocalLlamaBackend],
    root_out: Path,
    split_hashes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if method in {"bapr_repair_k_anchor", "bapr_si_k_anchor"}:
        if k is None:
            raise ValueError(f"{method} requires an explicit k value")
        return run_bapr_one(
            method=method,
            k=int(k),
            config=config,
            fold=fold,
            seed=seed,
            train=train,
            val=val,
            test=test,
            instruction=instruction,
            backend=backend,
            root_out=root_out,
            split_hashes=split_hashes,
        )

    score_min = int(config["data"]["score_min"])
    score_max = int(config["data"]["score_max"])
    run_name = run_name_for(method, k)
    out_dir = root_out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    selection_val = val
    if method in {"sisa_k_anchor", "coverage_first_sisa_k_anchor"}:
        val_diag, val_sel, split_meta = split_val_diag_sel(
            val,
            score_min,
            score_max,
            float(config.get("sisa", config.get("bapr", {})).get("val_diag_ratio", config.get("bapr", {}).get("val_diag_ratio", 0.5))),
        )
        selection_val = val_diag
        write_json(out_dir / ("coverage_first_sisa_val_split.json" if method == "coverage_first_sisa_k_anchor" else "sisa_val_split.json"), split_meta)
    anchors, trace, rep_changed, features = build_anchor_bank(
        method, k, train, selection_val, score_min, score_max, seed, config, instruction, backend, out_dir
    )
    trace_path = out_dir / "anchor_selection_trace.jsonl"
    write_jsonl(trace_path, trace)
    if method in {
        "representation_guided_k_anchor",
        "stratified_rep_guided_k_anchor",
        "retrieval_grounded_stratified_rep_k_anchor",
        "retrieval_grounded_stratified_rep_k_anchor_v21",
        "retrieval_grounded_no_rep_k_anchor",
        "sisa_k_anchor",
        "coverage_first_sisa_k_anchor",
    }:
        write_jsonl(out_dir / "representation_selection_trace.jsonl", trace)
        write_csv(out_dir / "representation_anchor_scores.csv", trace)
    if method == "stability_retrieval_k_anchor":
        write_jsonl(out_dir / "stability_selection_trace.jsonl", trace)
    bank = AnchorBank(
        anchor_bank_id=stable_hash({"method": method, "k": k, "anchors": [a.essay_id for a in anchors]}),
        method=method,
        k=int(k or len(anchors)),
        anchor_ids=[a.essay_id for a in anchors],
        score_coverage=score_coverage(anchors, score_min, score_max),
        token_cost=sum(a.token_length for a in anchors),
        score_range=[score_min, score_max],
        selection_trace_path=str(trace_path),
        representation_changed_anchor_choice=rep_changed,
        representation_features_used=features,
    )
    bank_payload = asdict(bank)
    if method == "coverage_first_sisa_k_anchor":
        bank_payload["target_score_ladder"] = trace[0].get("target_score_ladder", []) if trace else []
        bank_payload["exact_score_coverage"] = {
            str(row.get("target_score")): bool(row.get("exact_score_match"))
            for row in trace
            if row.get("selected_reason") != "coverage_first_sisa: no candidate available"
        }
        bank_payload["coverage_gaps"] = {
            str(row.get("target_score")): int(row.get("coverage_gap", 0) or 0)
            for row in trace
            if row.get("target_score") is not None
        }
    write_json(out_dir / "anchor_bank.json", bank_payload)
    write_json(
        out_dir / "anchor_metrics.json",
        {
            **bank_payload,
            "anchor_count": len(anchors),
            "average_anchor_length": float(np.mean([a.token_length for a in anchors])) if anchors else 0.0,
            "selected_anchors": [asdict(a) for a in anchors],
            "anchor_replacement_count": 0,
        },
    )

    t0 = time.time()
    val_rows, val_usage = score_items(backend, val, instruction, anchors, score_min, score_max)
    test_rows, test_usage = score_items(backend, test, instruction, anchors, score_min, score_max)
    runtime = time.time() - t0
    pred_rows = []
    for split, rows in [("val", val_rows), ("test", test_rows)]:
        for row in rows:
            pred_rows.append({"split": split, **row})
    write_csv(out_dir / "predictions.csv", pred_rows)

    val_metrics = score_boundary_metrics(
        [r["gold_score"] for r in val_rows],
        [r["pred_score"] for r in val_rows],
        score_min,
        score_max,
    )
    test_metrics = score_boundary_metrics(
        [r["gold_score"] for r in test_rows],
        [r["pred_score"] for r in test_rows],
        score_min,
        score_max,
    )
    write_json(out_dir / "score_boundary_metrics.json", {"val": val_metrics, "test": test_metrics})
    dist_rows = []
    for split, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for score in range(score_min, score_max + 1):
            dist_rows.append(
                {
                    "split": split,
                    "score": score,
                    "gold_count": metrics["gold_distribution"].get(str(score), 0),
                    "pred_count": metrics["prediction_distribution"].get(str(score), 0),
                    "per_score_recall": metrics["per_score_recall"].get(str(score)),
                }
            )
    write_csv(out_dir / "prediction_distribution.csv", dist_rows)
    usage = {k: int(val_usage.get(k, 0) + test_usage.get(k, 0)) for k in set(val_usage) | set(test_usage)}
    total_tokens = int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0) + usage.get("representation_tokens", 0))
    summary = {
        "method": method,
        "k": int(k or len(anchors)),
        "exp_dir": str(out_dir),
        "val_qwk": val_metrics["qwk"],
        "test_qwk": test_metrics["qwk"],
        "mae": test_metrics["mae"],
        "high_recall": test_metrics["high_recall"],
        "max_recall": test_metrics["max_recall"],
        "SCI": test_metrics["score_compression_index"],
        "range_coverage": test_metrics["range_coverage"],
        "score_TV": test_metrics["score_tv"],
        "tokens": total_tokens,
        "runtime_sec": runtime,
        "anchor_count": len(anchors),
        "anchor_token_cost": bank.token_cost,
        "representation_changed_anchor_choice": rep_changed,
        "representation_features_used": features,
        "resumed_from_existing_outputs": False,
    }
    write_json(out_dir / "run_summary.json", summary)
    md = [
        "# Summary\n\n",
        "## Goal\nBudgeted anchor protocol baseline for raw LLM essay scoring.\n\n",
        "## Setup\n",
        f"- method: {method}\n- k: {summary['k']}\n- final_pace_calibrated: false\n- test-time calibration: none\n\n",
        "## Results\n",
        f"- val QWK: {summary['val_qwk']:.4f}\n- raw test QWK: {summary['test_qwk']:.4f}\n",
        f"- MAE: {summary['mae']:.4f}\n- high recall: {summary['high_recall']:.4f}\n",
        f"- max recall: {summary['max_recall']:.4f}\n- SCI: {summary['SCI']:.4f}\n",
        f"- range coverage: {summary['range_coverage']:.4f}\n- score TV: {summary['score_TV']:.4f}\n\n",
        "## Anchor Analysis\n",
        f"- selected anchor ids: {bank.anchor_ids}\n- score coverage: {bank.score_coverage}\n",
        f"- token cost: {bank.token_cost}\n- representation changed choice: {rep_changed}\n\n",
    ]
    (out_dir / "summary.md").write_text("".join(md), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anchor_budget_phase1_p1.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--ks", nargs="*", type=int, default=None)
    parser.add_argument("--output-root", default=None, help="Reuse an existing experiment root and resume incomplete runs.")
    parser.add_argument("--split_manifest", default=None, help="Use fixed train/val/test essay IDs from a split manifest JSON.")
    parser.add_argument("--no-resume-existing", action="store_true", help="Recompute runs even if required outputs already exist.")
    parser.add_argument("--fake-scoring", action="store_true", help="Use deterministic fake scores and do not load the local LLM.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.fake_scoring:
        config["fake_scoring"] = True
    seed = int(config.get("debug", {}).get("seed", 42 + args.fold))
    random.seed(seed)
    np.random.seed(seed)
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else None
    train, val, test = load_asap_split(config, args.fold, split_manifest=split_manifest_path)
    split_hashes = split_hash_summary(train, val, test)
    if split_manifest_path is not None:
        config["fixed_split_manifest"] = {
            "path": str(split_manifest_path),
            **split_hashes,
        }
    instruction = instruction_from_config(config)
    methods = args.methods or list(config.get("anchor_budget", {}).get("methods", []))
    ks = args.ks or list(config.get("anchor_budget", {}).get("k_values", [3, 6, 9]))
    if args.output_root:
        out_root = Path(args.output_root)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_root = Path(config.get("output", {}).get("root", "logs/anchor_budget")) / f"phase1_p{config['data']['essay_set']}_fold{args.fold}_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "config.yaml.json", config)
    with open(out_root / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    plan = []
    for method in methods:
        if method in {"no_anchor", "full_static_anchor"}:
            plan.append((method, None))
        else:
            for k in ks:
                plan.append((method, int(k)))
    if args.dry_run:
        print(json.dumps({"output_root": str(out_root), "plan": plan, "split": split_hashes}, ensure_ascii=False, indent=2))
        return

    if args.fake_scoring:
        backend: Optional[LocalLlamaBackend] = None
        print("[AnchorBudget] Fake scoring enabled; local LLM will not be loaded.")
    else:
        backend = LocalLlamaBackend(
            config=config,
            model_path=config.get("pace", {}).get("model_path", config.get("model", {}).get("name")),
            dtype=config.get("pace", {}).get("dtype", "bfloat16"),
            load_in_4bit=bool(config.get("pace", {}).get("load_in_4bit", False)),
        )
    summaries = []
    for method, k in plan:
        run_dir = out_root / run_name_for(method, k)
        if not args.no_resume_existing and is_run_complete(run_dir, method):
            print(f"[AnchorBudget] Reusing completed {method} k={k} from {run_dir}")
            summaries.append(load_existing_run_summary(run_dir, method, k))
            continue
        print(f"[AnchorBudget] Running {method} k={k}")
        summaries.append(
            run_one(
                method=method,
                k=k,
                config=config,
                fold=args.fold,
                seed=seed,
                train=train,
                val=val,
                test=test,
                instruction=instruction,
                backend=backend,
                root_out=out_root,
                split_hashes=split_hashes,
            )
        )
    write_csv(out_root / "phase1_comparison_table.csv", summaries)
    write_curve_files(out_root, summaries)
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    if any(row["method"] in {"bapr_repair_k_anchor", "bapr_si_k_anchor"} for row in summaries):
        decision = bapr_decision(summaries)
        summary_text = render_bapr_summary(config, args.fold, out_root, summaries, decision)
        (out_root / "bapr_v1_summary.md").write_text(summary_text, encoding="utf-8")
        (logs_dir / "bapr_v1_summary.md").write_text(summary_text, encoding="utf-8")
    else:
        decision = phase1_decision(summaries)
        summary_text = render_phase1_summary(config, args.fold, out_root, summaries, decision)
        (out_root / "new_mainline_phase1_anchor_budget_summary.md").write_text(summary_text, encoding="utf-8")
        (logs_dir / "new_mainline_phase1_anchor_budget_summary.md").write_text(summary_text, encoding="utf-8")
    print(f"[AnchorBudget] Output root: {out_root}")
    print(f"[AnchorBudget] Decision: {decision['decision']} - {decision['reason']}")


def write_curve_files(out_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    curve_specs = {
        "qwk_vs_k.csv": "test_qwk",
        "high_recall_vs_k.csv": "high_recall",
        "sci_vs_k.csv": "SCI",
        "range_coverage_vs_k.csv": "range_coverage",
        "token_cost_vs_qwk.csv": "tokens",
    }
    for filename, metric in curve_specs.items():
        curve_rows = []
        for row in rows:
            curve_rows.append(
                {
                    "method": row["method"],
                    "k": row["k"],
                    "metric": metric,
                    "value": row.get(metric),
                    "test_qwk": row.get("test_qwk"),
                    "tokens": row.get("tokens"),
                }
            )
        write_csv(out_root / filename, curve_rows)


def phase1_decision(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    by = {(r["method"], int(r["k"])): r for r in rows}
    support = []
    for k in sorted({int(r["k"]) for r in rows if r["method"] != "full_static_anchor"}):
        rep = by.get(("representation_guided_k_anchor", k))
        if not rep:
            continue
        for baseline in ["static_k_anchor", "stratified_k_anchor", "retrieval_k_anchor"]:
            base = by.get((baseline, k))
            if base and (
                rep["val_qwk"] > base["val_qwk"]
                or rep["high_recall"] > base["high_recall"]
                or rep["SCI"] > base["SCI"]
                or rep["range_coverage"] > base["range_coverage"]
            ):
                support.append(f"rep k={k} improves over {baseline}")
    changed = any(r.get("representation_changed_anchor_choice") for r in rows if r["method"] == "representation_guided_k_anchor")
    if support and changed:
        return {"decision": "PASS", "reason": "; ".join(support[:4])}
    if support:
        return {"decision": "INCONCLUSIVE", "reason": "some metric improved, but representation did not change anchor choice"}
    return {"decision": "FAIL", "reason": "representation-guided anchors did not beat same-budget baselines"}


def bapr_decision(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    bapr_rows = [row for row in rows if row["method"] in {"bapr_repair_k_anchor", "bapr_si_k_anchor"}]
    if not bapr_rows:
        return {"decision": "NO_BAPR_RUNS", "reason": "no BAPR method rows were found"}
    reasons = []
    for row in bapr_rows:
        selected = row.get("bapr_selected_reason", "")
        reasons.append(f"k={row.get('k')}: {selected or 'completed'}")
    return {
        "decision": "BAPR_RUN_COMPLETE",
        "reason": "; ".join(reasons),
    }


def render_phase1_summary(config: Dict[str, Any], fold: int, out_root: Path, rows: Sequence[Dict[str, Any]], decision: Dict[str, str]) -> str:
    commit = os.environ.get("WISE_PACE_COMMIT", "")
    lines = [
        "# Summary\n\n",
        "## Goal\n",
        "Test whether representation-guided budgeted anchor protocol selection improves raw LLM essay scoring under a fixed local model.\n\n",
        "## Setup\n",
        f"- commit: `{commit}`\n",
        f"- model: `{config.get('model', {}).get('name')}`\n",
        f"- prompt / essay set: {config['data']['essay_set']}\n",
        f"- fold: {fold}\n",
        f"- seed: {config.get('debug', {}).get('seed')}\n",
        f"- split_manifest: `{config.get('fixed_split_manifest', {}).get('path', '')}`\n",
        f"- train_ids_hash: `{config.get('fixed_split_manifest', {}).get('train_ids_hash', '')}`\n",
        f"- val_ids_hash: `{config.get('fixed_split_manifest', {}).get('val_ids_hash', '')}`\n",
        f"- test_ids_hash: `{config.get('fixed_split_manifest', {}).get('test_ids_hash', '')}`\n",
        f"- methods: {config.get('anchor_budget', {}).get('methods')}\n",
        f"- k values: {config.get('anchor_budget', {}).get('k_values')}\n",
        "- anchor pool: train split only\n",
        "- final_pace_calibrated: false\n",
        "- parser: LocalLlamaBackend raw JSON score parser\n\n",
        "## Results Table\n",
        "| method | k | val_qwk | test_qwk | mae | high_recall | max_recall | SCI | range_coverage | score_TV | tokens | runtime_sec |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['k']} | {r['val_qwk']:.4f} | {r['test_qwk']:.4f} | "
            f"{r['mae']:.4f} | {r['high_recall']:.4f} | {r['max_recall']:.4f} | "
            f"{r['SCI']:.4f} | {r['range_coverage']:.4f} | {r['score_TV']:.4f} | "
            f"{r['tokens']} | {r['runtime_sec']:.1f} |\n"
        )
    lines.extend(
        [
            "\n## Anchor Analysis\n",
            f"- combined output root: `{out_root}`\n",
            "- each run contains `anchor_bank.json`, `anchor_selection_trace.jsonl`, `anchor_metrics.json`, predictions, boundary metrics, and summary.\n",
            "- representation-guided runs record `representation_anchor_scores.csv` and `representation_selection_trace.jsonl`.\n\n",
            "## Score-Boundary Analysis\n",
            "- See each run's `score_boundary_metrics.json` and `prediction_distribution.csv` for compression, range usage, per-score recall, and high-tail under-scoring.\n\n",
            "## Decision\n",
            f"- {decision['decision']}: {decision['reason']}\n",
            f"- Can proceed: {'yes' if decision['decision'] == 'PASS' else 'no'}\n",
        ]
    )
    return "".join(lines)


def render_bapr_summary(config: Dict[str, Any], fold: int, out_root: Path, rows: Sequence[Dict[str, Any]], decision: Dict[str, str]) -> str:
    commit = os.environ.get("WISE_PACE_COMMIT", "")
    lines = [
        "# BAPR v1 Summary\n\n",
        "## Goal\n",
        "Run boundary-aware one-step anchor-bank repair under a fixed instruction and fixed raw scoring model. "
        "This summary does not apply the old representation-guided Phase 1 pass/fail logic.\n\n",
        "## Setup\n",
        f"- commit: `{commit}`\n",
        f"- model: `{config.get('model', {}).get('name')}`\n",
        f"- prompt / essay set: {config['data']['essay_set']}\n",
        f"- fold: {fold}\n",
        f"- seed: {config.get('debug', {}).get('seed')}\n",
        f"- methods: {config.get('anchor_budget', {}).get('methods')}\n",
        f"- k values: {config.get('anchor_budget', {}).get('k_values')}\n",
        f"- fake_scoring: {bool(config.get('fake_scoring', False))}\n",
        "- anchor pool: train split only; V_sel and test are excluded from repair candidates\n",
        "- final_pace_calibrated: false\n",
        "- test-time calibration: none\n\n",
        "## Results Table\n",
        "| method | k | val_sel_qwk | test_qwk | mae | high_recall | max_recall | SCI | range_coverage | score_TV | selected_reason | tokens | runtime_sec |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|\n",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['k']} | {r['val_qwk']:.4f} | {r['test_qwk']:.4f} | "
            f"{r['mae']:.4f} | {r['high_recall']:.4f} | {r['max_recall']:.4f} | "
            f"{r['SCI']:.4f} | {r['range_coverage']:.4f} | {r['score_TV']:.4f} | "
            f"{r.get('bapr_selected_reason', '')} | {r['tokens']} | {r['runtime_sec']:.1f} |\n"
        )
    lines.extend(
        [
            "\n## Required Outputs\n",
            f"- combined output root: `{out_root}`\n",
            "- each BAPR run writes `bapr_parent_anchor_bank.json`, `bapr_parent_metrics.json`, "
            "`bapr_failure_profile.json`, `bapr_repair_candidates.jsonl`, "
            "`bapr_guarded_selection.csv`, `bapr_final_anchor_bank.json`, and `bapr_repair_trace.jsonl`.\n\n",
            "## Decision\n",
            f"- {decision['decision']}: {decision['reason']}\n",
            "- This is a pipeline validity decision only; it is not a full-fold or main-claim decision.\n",
        ]
    )
    return "".join(lines)


if __name__ == "__main__":
    main()
