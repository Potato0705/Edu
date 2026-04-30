"""PACE-AES evidence construction.

This module implements the first runnable slice of the method skeleton:

* anchor-conditioned residual embeddings r_emb
* lightweight structured-reasoning features r_s
* objective essay features f_o
* uncertainty / hedge features u
* final evidence vector z = [y_raw ; r_emb ; r_s ; f_o ; u]

The goal is not to perfectly realize every feature proposed in the draft, but
to provide a faithful, extensible implementation that turns Layer-1 protocol
outputs into a trainable Layer-2/3 calibration input.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from pace.llm_backend import AnchorCacheEntry, ScoringResult


_RE_WORD = re.compile(r"\b\w+\b", re.UNICODE)
_RE_SENT_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")
_RE_SCORE_INT = re.compile(r"-?\d+")

_HEDGE_WORDS = {
    "somewhat",
    "partially",
    "may",
    "might",
    "could",
    "appears",
    "seems",
    "arguably",
    "relatively",
    "limited",
    "uneven",
    "adequate",
}
_TRAIT_WORDS = {
    "coherence",
    "organization",
    "grammar",
    "support",
    "evidence",
    "development",
    "focus",
    "clarity",
    "transition",
    "vocabulary",
}
_RISK_WORDS = {
    "off-topic",
    "weak",
    "limited",
    "insufficient",
    "unclear",
    "confusing",
    "inconsistent",
    "underdeveloped",
}
_STRENGTH_WORDS = {
    "clear",
    "coherent",
    "organized",
    "strong",
    "effective",
    "well-developed",
    "engaging",
    "focused",
}


@dataclass
class EvidenceBundle:
    z: torch.Tensor
    r_emb: torch.Tensor
    r_s: torch.Tensor
    f_o: torch.Tensor
    u: torch.Tensor
    y_raw: int
    meta: Dict


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _tokenize_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in _RE_WORD.finditer(text)]


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _RE_SENT_SPLIT.split(text) if s.strip()]
    if not parts and text.strip():
        return [text.strip()]
    return parts


def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not parts and text.strip():
        return [text.strip()]
    return parts


def _count_keyword_hits(text: str, vocab: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term) for term in vocab)


def _extract_in_range_score_mentions(raw_text: str, score_min: int, score_max: int) -> List[int]:
    vals: List[int] = []
    for tok in _RE_SCORE_INT.findall(raw_text):
        try:
            val = int(tok)
        except ValueError:
            continue
        if score_min <= val <= score_max:
            vals.append(val)
    return vals


def build_anchor_residual_features(
    hidden: torch.Tensor,
    anchor_hidden: torch.Tensor,
) -> torch.Tensor:
    """Return [h(x); Δ_low; Δ_mid; Δ_high; cosines; L2 norms]."""
    if hidden.ndim != 1:
        raise ValueError(f"Expected hidden shape (d,), got {tuple(hidden.shape)}")
    if anchor_hidden.ndim != 2 or anchor_hidden.shape[0] < 3:
        raise ValueError(
            f"Expected anchor_hidden shape (n>=3, d), got {tuple(anchor_hidden.shape)}"
        )
    h = hidden.float()
    anchors = anchor_hidden.float()
    deltas = h.unsqueeze(0) - anchors
    cosines = F.cosine_similarity(h.unsqueeze(0), anchors, dim=1)
    l2 = deltas.norm(p=2, dim=1)
    return torch.cat([h, deltas.reshape(-1), cosines, l2], dim=0)


def build_reasoning_features(raw_text: str, score_min: int, score_max: int) -> torch.Tensor:
    words = _tokenize_words(raw_text)
    n_words = len(words)
    score_mentions = _extract_in_range_score_mentions(raw_text, score_min, score_max)
    score_span = (max(score_mentions) - min(score_mentions)) if score_mentions else 0
    score_mean = float(sum(score_mentions)) / len(score_mentions) if score_mentions else 0.0
    score_std = (
        math.sqrt(sum((x - score_mean) ** 2 for x in score_mentions) / len(score_mentions))
        if score_mentions
        else 0.0
    )
    feats = [
        math.log1p(n_words),
        float("final_score" in raw_text),
        _safe_div(_count_keyword_hits(raw_text, _TRAIT_WORDS), max(1, n_words)),
        _safe_div(_count_keyword_hits(raw_text, _STRENGTH_WORDS), max(1, n_words)),
        _safe_div(_count_keyword_hits(raw_text, _RISK_WORDS), max(1, n_words)),
        _safe_div(raw_text.lower().count("compared"), max(1, n_words)),
        _safe_div(raw_text.lower().count("example"), max(1, n_words)),
        _safe_div(raw_text.lower().count("anchor"), max(1, n_words)),
        float(len(score_mentions)),
        float(score_span),
        float(score_std),
    ]
    return torch.tensor(feats, dtype=torch.float32)


def build_objective_features(essay_text: str) -> torch.Tensor:
    words = _tokenize_words(essay_text)
    sents = _split_sentences(essay_text)
    paras = _split_paragraphs(essay_text)
    word_lens = [len(w) for w in words]
    sent_word_counts = [len(_tokenize_words(s)) for s in sents] or [0]
    para_word_counts = [len(_tokenize_words(p)) for p in paras] or [0]
    uniq = len(set(words))
    punctuation_count = sum(1 for ch in essay_text if ch in ",.;:!?")
    digit_count = sum(1 for ch in essay_text if ch.isdigit())
    long_word_count = sum(1 for w in words if len(w) >= 7)

    mean_sent = float(sum(sent_word_counts)) / len(sent_word_counts)
    std_sent = (
        math.sqrt(
            sum((x - mean_sent) ** 2 for x in sent_word_counts) / len(sent_word_counts)
        )
        if sent_word_counts
        else 0.0
    )
    mean_para = float(sum(para_word_counts)) / len(para_word_counts)

    feats = [
        math.log1p(len(words)),
        math.log1p(len(sents)),
        math.log1p(len(paras)),
        mean_sent,
        std_sent,
        mean_para,
        _safe_div(uniq, max(1, len(words))),
        _safe_div(long_word_count, max(1, len(words))),
        _safe_div(sum(word_lens), max(1, len(words))),
        _safe_div(punctuation_count, max(1, len(words))),
        _safe_div(digit_count, max(1, len(words))),
    ]
    return torch.tensor(feats, dtype=torch.float32)


def build_uncertainty_features(raw_text: str, y_raw: int, score_min: int, score_max: int) -> torch.Tensor:
    words = _tokenize_words(raw_text)
    n_words = len(words)
    score_mentions = _extract_in_range_score_mentions(raw_text, score_min, score_max)
    raw_norm = _safe_div(y_raw - score_min, max(1, score_max - score_min))
    mention_mean = (
        float(sum(score_mentions)) / len(score_mentions) if score_mentions else float(y_raw)
    )
    mention_std = (
        math.sqrt(sum((x - mention_mean) ** 2 for x in score_mentions) / len(score_mentions))
        if score_mentions
        else 0.0
    )
    feats = [
        _safe_div(_count_keyword_hits(raw_text, _HEDGE_WORDS), max(1, n_words)),
        _safe_div(raw_text.count("?"), max(1, len(raw_text))),
        float(len(score_mentions)),
        float(mention_std),
        float(score_mentions[-1] - score_mentions[0]) if len(score_mentions) >= 2 else 0.0,
        raw_norm,
    ]
    return torch.tensor(feats, dtype=torch.float32)


def build_evidence_vector(
    *,
    essay_text: str,
    result: ScoringResult,
    anchor_entry: AnchorCacheEntry,
    score_min: int,
    score_max: int,
) -> EvidenceBundle:
    if result.hidden is None:
        raise RuntimeError("ScoringResult.hidden is required for PACE evidence.")
    raw_score_norm = _safe_div(result.y_raw - score_min, max(1, score_max - score_min))
    raw_score_feat = torch.tensor([raw_score_norm], dtype=torch.float32)
    r_emb = build_anchor_residual_features(result.hidden, anchor_entry.hidden)
    r_s = build_reasoning_features(result.raw_text, score_min, score_max)
    f_o = build_objective_features(essay_text)
    u = build_uncertainty_features(result.raw_text, result.y_raw, score_min, score_max)
    z = torch.cat([raw_score_feat, r_emb, r_s, f_o, u], dim=0).float()
    return EvidenceBundle(
        z=z,
        r_emb=r_emb,
        r_s=r_s,
        f_o=f_o,
        u=u,
        y_raw=result.y_raw,
        meta={
            "raw_score_norm": raw_score_norm,
            "hidden_dim": int(result.hidden.numel()),
            "r_emb_dim": int(r_emb.numel()),
            "r_s_dim": int(r_s.numel()),
            "f_o_dim": int(f_o.numel()),
            "u_dim": int(u.numel()),
        },
    )


__all__ = [
    "EvidenceBundle",
    "build_anchor_residual_features",
    "build_reasoning_features",
    "build_objective_features",
    "build_uncertainty_features",
    "build_evidence_vector",
]
