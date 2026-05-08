"""Neural hidden-evidence encoder for WISE-PACE.

This module is intentionally free of hand-crafted essay statistics. It consumes
hidden states produced by the local scoring model and returns a compact neural
evidence vector that can be used by the lightweight PACE probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeuralEvidenceConfig:
    hidden_input_dim: int
    proj_dim: int = 128
    score_emb_dim: int = 32
    output_dim: int = 128
    dropout: float = 0.1
    anchor_view: str = "dual"  # none/text/score/dual
    use_reasoning_hidden: bool = True
    use_score_distribution: bool = False
    use_raw_score_embedding: bool = True
    num_attention_heads: int = 2
    attention_mode: str = "cosine"  # cosine/mha
    preserve_block_structure: bool = True
    projection_seed: int = 42

    def __post_init__(self) -> None:
        allowed = {"none", "text", "score", "dual"}
        if self.anchor_view not in allowed:
            raise ValueError(f"anchor_view must be one of {sorted(allowed)}, got {self.anchor_view!r}")
        allowed_attention = {"cosine", "mha"}
        if self.attention_mode not in allowed_attention:
            raise ValueError(
                f"attention_mode must be one of {sorted(allowed_attention)}, got {self.attention_mode!r}"
            )
        self.proj_dim = int(self.proj_dim)
        self.score_emb_dim = int(self.score_emb_dim)
        self.output_dim = int(self.output_dim)
        self.hidden_input_dim = int(self.hidden_input_dim)
        if self.output_dim < 4:
            raise ValueError("output_dim must be at least 4 for block-preserving neural evidence")
        self.num_attention_heads = max(1, int(self.num_attention_heads))
        if self.proj_dim % self.num_attention_heads != 0:
            self.num_attention_heads = 1
        self.projection_seed = int(self.projection_seed)


class NeuralEvidenceEncoder(nn.Module):
    """Encode model-hidden evidence into a fixed-dimensional vector.

    Inputs are hidden states, optional dual-view anchor representations, optional
    reasoning hidden states, and a raw score or future score distribution. No
    lexical counts, readability features, or keyword features are used here.
    """

    def __init__(self, config: NeuralEvidenceConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_input_dim)
        p = int(config.proj_dim)
        s = int(config.score_emb_dim)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config.projection_seed))
            self.target_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, p), nn.GELU())
            self.anchor_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, p), nn.GELU())
            self.reasoning_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, p), nn.GELU())
            self.anchor_score_proj = nn.Sequential(nn.Linear(1, p), nn.GELU(), nn.Linear(p, p))
            self.raw_score_proj = nn.Sequential(nn.Linear(1, s), nn.GELU(), nn.Linear(s, s))
            self.score_dist_proj = nn.Sequential(nn.Linear(3, s), nn.GELU(), nn.Linear(s, s))
            self.anchor_attention = nn.MultiheadAttention(
                embed_dim=p,
                num_heads=int(config.num_attention_heads),
                dropout=float(config.dropout),
                batch_first=True,
            )
            target_dim, anchor_dim, reasoning_dim, score_dim = self.block_dims()
            self.target_head = nn.Sequential(nn.LayerNorm(p), nn.Linear(p, target_dim))
            self.anchor_head = nn.Sequential(nn.LayerNorm(p), nn.Linear(p, anchor_dim))
            self.reasoning_head = nn.Sequential(nn.LayerNorm(p), nn.Linear(p, reasoning_dim))
            self.score_head = nn.Sequential(nn.LayerNorm(s), nn.Linear(s, score_dim))
            in_dim = p + p + p + s
            self.output_mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, max(p, config.output_dim)),
                nn.GELU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(max(p, config.output_dim), int(config.output_dim)),
            )
        self.eval()

    def block_dims(self) -> Tuple[int, int, int, int]:
        """Return output block sizes: target, anchor, reasoning, score."""
        dim = int(self.config.output_dim)
        target = max(1, dim // 4)
        anchor = max(1, dim // 4)
        reasoning = max(1, dim // 4)
        score = dim - target - anchor - reasoning
        if score < 1:
            raise ValueError("output_dim is too small for four neural evidence blocks")
        return target, anchor, reasoning, score

    def forward(
        self,
        *,
        target_hidden: torch.Tensor,
        anchor_hidden_text: Optional[torch.Tensor] = None,
        anchor_hidden_score: Optional[torch.Tensor] = None,
        anchor_scores: Optional[torch.Tensor] = None,
        score_min: int,
        score_max: int,
        reasoning_hidden: Optional[torch.Tensor] = None,
        score_probs: Optional[torch.Tensor] = None,
        y_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | Dict[str, object]]]:
        if target_hidden.dim() == 1:
            target_hidden = target_hidden.unsqueeze(0)
        target_hidden = target_hidden.float()
        batch_size = int(target_hidden.shape[0])
        device = target_hidden.device

        target_proj = self.target_proj(target_hidden)
        anchor_tokens = self._build_anchor_tokens(
            batch_size=batch_size,
            device=device,
            anchor_hidden_text=anchor_hidden_text,
            anchor_hidden_score=anchor_hidden_score,
            anchor_scores=anchor_scores,
            score_min=score_min,
            score_max=score_max,
        )
        if anchor_tokens is None or anchor_tokens.shape[1] == 0:
            anchor_context = torch.zeros(batch_size, self.config.proj_dim, device=device)
            attention_weights = torch.zeros(batch_size, 0, device=device)
        elif self.config.attention_mode == "cosine":
            sim = F.cosine_similarity(target_proj.unsqueeze(1), anchor_tokens, dim=-1)
            attention_weights = torch.softmax(sim, dim=-1)
            anchor_context = torch.bmm(attention_weights.unsqueeze(1), anchor_tokens)[:, 0, :]
        else:
            attended, attn = self.anchor_attention(
                query=target_proj.unsqueeze(1),
                key=anchor_tokens,
                value=anchor_tokens,
                need_weights=True,
            )
            anchor_context = attended[:, 0, :]
            attention_weights = attn[:, 0, :]

        if self.config.use_reasoning_hidden and reasoning_hidden is not None:
            if reasoning_hidden.dim() == 1:
                reasoning_hidden = reasoning_hidden.unsqueeze(0)
            reasoning_hidden = reasoning_hidden.float().to(device)
            if reasoning_hidden.shape[0] == 1 and batch_size > 1:
                reasoning_hidden = reasoning_hidden.expand(batch_size, -1)
            reasoning_proj = self.reasoning_proj(reasoning_hidden)
        else:
            reasoning_proj = torch.zeros(batch_size, self.config.proj_dim, device=device)

        score_emb = self._build_score_embedding(
            batch_size=batch_size,
            device=device,
            score_min=score_min,
            score_max=score_max,
            y_raw=y_raw,
            score_probs=score_probs,
        )
        if self.config.preserve_block_structure:
            z = torch.cat(
                [
                    self.target_head(target_proj),
                    self.anchor_head(anchor_context),
                    self.reasoning_head(reasoning_proj),
                    self.score_head(score_emb),
                ],
                dim=-1,
            )
        else:
            z = self.output_mlp(torch.cat([target_proj, anchor_context, reasoning_proj, score_emb], dim=-1))
        aux: Dict[str, torch.Tensor | Dict[str, object]] = {
            "attention_weights": attention_weights.detach(),
            "anchor_context": anchor_context.detach(),
            "enabled_views": {
                "anchor_view": self.config.anchor_view,
                "attention_mode": self.config.attention_mode,
                "preserve_block_structure": self.config.preserve_block_structure,
                "use_reasoning_hidden": self.config.use_reasoning_hidden,
                "use_raw_score_embedding": self.config.use_raw_score_embedding,
                "use_score_distribution": self.config.use_score_distribution,
            },
        }
        return z, aux

    def _build_anchor_tokens(
        self,
        *,
        batch_size: int,
        device: torch.device,
        anchor_hidden_text: Optional[torch.Tensor],
        anchor_hidden_score: Optional[torch.Tensor],
        anchor_scores: Optional[torch.Tensor],
        score_min: int,
        score_max: int,
    ) -> Optional[torch.Tensor]:
        view = self.config.anchor_view
        if view == "none":
            return None

        tokens = []
        if view in {"text", "dual"} and anchor_hidden_text is not None:
            text_h = self._expand_anchor_hidden(anchor_hidden_text, batch_size, device)
            tokens.append(self.anchor_proj(text_h))
        if view in {"score", "dual"} and anchor_hidden_score is not None:
            score_h = self._expand_anchor_hidden(anchor_hidden_score, batch_size, device)
            score_tokens = self.anchor_proj(score_h)
            score_tokens = score_tokens + self._anchor_score_embedding(
                anchor_scores,
                batch_size=batch_size,
                n_anchors=int(score_tokens.shape[1]),
                device=device,
                score_min=score_min,
                score_max=score_max,
            )
            tokens.append(score_tokens)
        if not tokens:
            return None
        return torch.cat(tokens, dim=1)

    def _expand_anchor_hidden(
        self,
        anchor_hidden: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        anchor_hidden = anchor_hidden.float().to(device)
        if anchor_hidden.dim() == 2:
            anchor_hidden = anchor_hidden.unsqueeze(0).expand(batch_size, -1, -1)
        elif anchor_hidden.dim() == 3 and anchor_hidden.shape[0] == 1 and batch_size > 1:
            anchor_hidden = anchor_hidden.expand(batch_size, -1, -1)
        if anchor_hidden.dim() != 3:
            raise ValueError(f"anchor_hidden must be [A,H] or [B,A,H], got {tuple(anchor_hidden.shape)}")
        return anchor_hidden

    def _anchor_score_embedding(
        self,
        anchor_scores: Optional[torch.Tensor],
        *,
        batch_size: int,
        n_anchors: int,
        device: torch.device,
        score_min: int,
        score_max: int,
    ) -> torch.Tensor:
        if anchor_scores is None:
            return torch.zeros(batch_size, n_anchors, self.config.proj_dim, device=device)
        scores = anchor_scores.float().to(device)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0).expand(batch_size, -1)
        elif scores.dim() == 2 and scores.shape[0] == 1 and batch_size > 1:
            scores = scores.expand(batch_size, -1)
        scores = scores[:, :n_anchors]
        if scores.shape[1] < n_anchors:
            pad = torch.full(
                (batch_size, n_anchors - scores.shape[1]),
                float(score_min),
                device=device,
            )
            scores = torch.cat([scores, pad], dim=1)
        span = max(1.0, float(score_max - score_min))
        scores_norm = ((scores - float(score_min)) / span).clamp(0.0, 1.0).unsqueeze(-1)
        return self.anchor_score_proj(scores_norm)

    def _build_score_embedding(
        self,
        *,
        batch_size: int,
        device: torch.device,
        score_min: int,
        score_max: int,
        y_raw: Optional[torch.Tensor],
        score_probs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pieces = []
        if self.config.use_raw_score_embedding and y_raw is not None:
            y = y_raw.float().to(device)
            if y.dim() == 0:
                y = y.view(1)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            if y.shape[0] == 1 and batch_size > 1:
                y = y.expand(batch_size, -1)
            span = max(1.0, float(score_max - score_min))
            y_norm = ((y[:, :1] - float(score_min)) / span).clamp(0.0, 1.0)
            pieces.append(self.raw_score_proj(y_norm))
        if self.config.use_score_distribution and score_probs is not None:
            probs = score_probs.float().to(device)
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            if probs.shape[0] == 1 and batch_size > 1:
                probs = probs.expand(batch_size, -1)
            probs = probs.clamp_min(1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            positions = torch.linspace(0.0, 1.0, probs.shape[-1], device=device)
            expected = (probs * positions).sum(dim=-1, keepdim=True)
            entropy = -(probs * probs.log()).sum(dim=-1, keepdim=True)
            entropy = entropy / max(1.0, float(torch.log(torch.tensor(probs.shape[-1], device=device))))
            max_prob = probs.max(dim=-1, keepdim=True).values
            pieces.append(self.score_dist_proj(torch.cat([expected, entropy, max_prob], dim=-1)))
        if not pieces:
            return torch.zeros(batch_size, self.config.score_emb_dim, device=device)
        return torch.stack(pieces, dim=0).mean(dim=0)
