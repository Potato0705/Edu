"""Prompt-level diagnostics for automatic PACE recipe selection."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


@dataclass
class LoadedEvidence:
    prompt_id: int
    fold: int
    score_min: int
    score_max: int
    num_anchors: int
    input_dim: int
    cache_path: Path
    payload: Dict
    splits: Dict[str, Dict]


def infer_prompt_fold_from_path(path: Path) -> tuple[int, int]:
    text = str(path).replace("\\", "/")
    match = re.search(r"p(\d+)_fold(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"asap_p(\d+)_fold(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Cannot infer prompt/fold from evidence path: {path}")


def infer_num_anchors_from_zdim(z_dim: int) -> int:
    # z = [raw(1); r_emb; r_s(11); f_o(11); u(6)]
    structured_dim = 1 + 11 + 11 + 6
    tail_free = z_dim - structured_dim
    for n_anchors in range(3, 9):
        numer = tail_free - (2 * n_anchors)
        denom = n_anchors + 1
        if numer > 0 and numer % denom == 0:
            hidden_dim = numer // denom
            if 256 <= hidden_dim <= 16384:
                return n_anchors
    return 3


def hydrate_split_aux(split_payload: Dict, *, prompt_id: int) -> None:
    z = split_payload["z"].float()
    if z.numel() == 0:
        split_payload["r_emb"] = torch.empty((0, 0), dtype=torch.float32)
        split_payload["u"] = torch.empty((0, 0), dtype=torch.float32)
        split_payload["uncertainty_scalar"] = torch.empty((0,), dtype=torch.float32)
        split_payload["prompt_id"] = torch.empty((0,), dtype=torch.long)
        return
    u_dim = 6
    fixed_tail = 11 + 11 + u_dim
    r_emb_dim = int(z.shape[1] - 1 - fixed_tail)
    if r_emb_dim <= 0:
        raise RuntimeError(f"Cannot infer r_emb_dim from z shape {tuple(z.shape)}")
    split_payload["r_emb"] = z[:, 1 : 1 + r_emb_dim]
    split_payload["u"] = z[:, -u_dim:]
    u_no_raw = split_payload["u"][:, :-1] if u_dim > 1 else split_payload["u"]
    split_payload["uncertainty_scalar"] = u_no_raw.norm(p=2, dim=1)
    split_payload["prompt_id"] = torch.full(
        (z.shape[0],),
        int(prompt_id),
        dtype=torch.long,
    )


def load_evidence_cache(
    path: Path,
    *,
    prompt_id: Optional[int] = None,
    fold: Optional[int] = None,
) -> LoadedEvidence:
    if prompt_id is None or fold is None:
        inferred_prompt, inferred_fold = infer_prompt_fold_from_path(path)
        prompt_id = inferred_prompt if prompt_id is None else prompt_id
        fold = inferred_fold if fold is None else fold
    payload = torch.load(path, map_location="cpu")
    splits = payload["splits"]
    meta = payload.get("meta", {})
    score_min = int(meta.get("score_min"))
    score_max = int(meta.get("score_max"))
    num_anchors = int(meta.get("num_anchors", 0))
    if num_anchors <= 0:
        num_anchors = infer_num_anchors_from_zdim(int(splits["train"]["z"].shape[1]))
    for split_name in ("train", "val", "test"):
        hydrate_split_aux(splits[split_name], prompt_id=prompt_id)
    return LoadedEvidence(
        prompt_id=int(prompt_id),
        fold=int(fold),
        score_min=score_min,
        score_max=score_max,
        num_anchors=num_anchors,
        input_dim=int(splits["train"]["z"].shape[1]),
        cache_path=path,
        payload=payload,
        splits=splits,
    )


def concat_splits(splits: Iterable[Dict]) -> Dict:
    parts = list(splits)
    rows: List[Dict] = []
    for payload in parts:
        rows.extend(list(payload.get("rows", [])))
    keys = [
        "essay_ids",
        "y_true",
        "y_raw",
        "z",
        "r_emb",
        "u",
        "uncertainty_scalar",
        "prompt_id",
    ]
    out = {"rows": rows}
    for key in keys:
        tensors = [p[key] for p in parts if key in p]
        out[key] = torch.cat(tensors, dim=0) if tensors else torch.empty(0)
    return out


def score_to_band_ids(
    scores: torch.Tensor,
    *,
    score_min: int,
    score_max: int,
    num_bands: int,
) -> torch.Tensor:
    span = max(1.0, float(score_max - score_min))
    frac = (scores.float() - float(score_min)) / span
    frac = frac.clamp(min=0.0, max=1.0)
    return torch.clamp((frac * float(num_bands)).long(), max=num_bands - 1)


def _safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return float("nan")


def localwise_diagnostics(
    split_payload: Dict,
    *,
    score_min: int,
    score_max: int,
    num_bands: int,
) -> Dict[str, float]:
    y_true = split_payload["y_true"].long()
    y_raw = split_payload["y_raw"].long()
    true_band = score_to_band_ids(
        y_true,
        score_min=score_min,
        score_max=score_max,
        num_bands=num_bands,
    )
    raw_band = score_to_band_ids(
        y_raw,
        score_min=score_min,
        score_max=score_max,
        num_bands=num_bands,
    )
    band_dist = (raw_band - true_band).abs().float()
    n = max(1, int(y_true.numel()))
    raw_float = y_raw.float()
    raw_counts = torch.bincount((y_raw - int(score_min)).clamp(min=0, max=int(score_max - score_min)))
    raw_mode_share = float(raw_counts.max().item() / n) if raw_counts.numel() else float("nan")
    raw_pred_std = float(raw_float.std(unbiased=False).item()) if n else float("nan")
    raw_pred_iqr = float(
        (torch.quantile(raw_float, 0.75) - torch.quantile(raw_float, 0.25)).item()
    ) if int(y_true.numel()) >= 4 else float("nan")
    return {
        "local_wise_qwk": _safe_qwk(y_true.numpy(), y_raw.numpy()),
        "cross_band_error_share": float((band_dist > 0).float().mean().item()) if n else float("nan"),
        "off_by_1_share": float((band_dist == 1).float().mean().item()) if n else float("nan"),
        "off_by_2plus_share": float((band_dist >= 2).float().mean().item()) if n else float("nan"),
        "mean_band_distance_errors": float(band_dist.mean().item()) if n else float("nan"),
        "raw_mode_share": raw_mode_share,
        "raw_pred_std": raw_pred_std,
        "raw_pred_iqr": raw_pred_iqr,
        "raw_std_frac": raw_pred_std / max(1.0, float(score_max - score_min)),
        "raw_iqr_frac": raw_pred_iqr / max(1.0, float(score_max - score_min)),
    }


def residual_overlap_diagnostics(
    split_payload: Dict,
    *,
    score_min: int,
    score_max: int,
    num_bands: int,
    min_samples_per_band: int,
    eps: float = 1.0e-8,
) -> Dict[str, float | str | int]:
    r_emb = split_payload["r_emb"].float()
    y_true = split_payload["y_true"].long()
    if r_emb.numel() == 0 or y_true.numel() == 0:
        return {
            "adjacent_overlap_score": float("nan"),
            "adjacent_overlap_reason": "empty_split",
            "adjacent_pairs_with_enough_samples": 0,
            "min_adjacent_pair_count": 0,
        }

    band_ids = score_to_band_ids(
        y_true,
        score_min=score_min,
        score_max=score_max,
        num_bands=num_bands,
    )
    centroids: Dict[int, torch.Tensor] = {}
    dispersions: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for band in range(num_bands):
        mask = band_ids == band
        count = int(mask.sum().item())
        counts[band] = count
        if count < min_samples_per_band:
            continue
        feats = r_emb[mask]
        centroid = feats.mean(dim=0)
        centroids[band] = centroid
        dispersions[band] = float((feats - centroid).pow(2).sum(dim=1).mean().item())

    overlaps: List[float] = []
    pair_counts: List[int] = []
    for band in range(num_bands - 1):
        if band not in centroids or (band + 1) not in centroids:
            continue
        gap = float((centroids[band] - centroids[band + 1]).pow(2).sum().item())
        overlap = (dispersions[band] + dispersions[band + 1]) / (gap + eps)
        if math.isfinite(overlap):
            overlaps.append(float(overlap))
            pair_counts.append(min(counts[band], counts[band + 1]))

    if not overlaps:
        return {
            "adjacent_overlap_score": float("nan"),
            "adjacent_overlap_reason": "insufficient_adjacent_band_samples",
            "adjacent_pairs_with_enough_samples": 0,
            "min_adjacent_pair_count": min(counts.values()) if counts else 0,
        }
    return {
        "adjacent_overlap_score": float(sum(overlaps) / len(overlaps)),
        "adjacent_overlap_reason": "ok",
        "adjacent_pairs_with_enough_samples": len(overlaps),
        "min_adjacent_pair_count": min(pair_counts) if pair_counts else 0,
    }


def uncertainty_diagnostics(split_payload: Dict) -> Dict[str, float]:
    unc = split_payload.get("uncertainty_scalar")
    if unc is None or unc.numel() == 0:
        return {
            "uncertainty_mean": float("nan"),
            "uncertainty_std": float("nan"),
            "high_uncertainty_ratio": float("nan"),
        }
    unc = unc.float()
    threshold = float(torch.quantile(unc, 0.75).item()) if unc.numel() >= 4 else float(unc.mean().item())
    return {
        "uncertainty_mean": float(unc.mean().item()),
        "uncertainty_std": float(unc.std(unbiased=False).item()),
        "high_uncertainty_ratio": float((unc >= threshold).float().mean().item()),
    }


def compute_prompt_diagnostics(
    evidence: LoadedEvidence,
    *,
    prompt: Optional[int] = None,
    fold: Optional[int] = None,
    min_adjacent_pair_count: int = 4,
) -> Dict[str, float | str | int]:
    prompt = evidence.prompt_id if prompt is None else int(prompt)
    fold = evidence.fold if fold is None else int(fold)
    train_val = concat_splits([evidence.splits["train"], evidence.splits["val"]])
    val = evidence.splits["val"]
    score_span = int(evidence.score_max - evidence.score_min)
    num_bands = int(evidence.num_anchors)
    avg_band_width = float(score_span) / max(1, num_bands)
    features: Dict[str, float | str | int] = {
        "prompt": int(prompt),
        "fold": int(fold),
        "score_min": int(evidence.score_min),
        "score_max": int(evidence.score_max),
        "score_span": int(score_span),
        "num_bands": int(num_bands),
        "avg_band_width": float(avg_band_width),
    }
    features.update(
        localwise_diagnostics(
            val,
            score_min=evidence.score_min,
            score_max=evidence.score_max,
            num_bands=num_bands,
        )
    )
    features.update(
        residual_overlap_diagnostics(
            train_val,
            score_min=evidence.score_min,
            score_max=evidence.score_max,
            num_bands=num_bands,
            min_samples_per_band=min_adjacent_pair_count,
        )
    )
    features.update(uncertainty_diagnostics(train_val))
    return features
