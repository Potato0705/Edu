"""Ordinal calibration for PACE-AES.

Implements:

* a compact CORAL-style residual calibrator
* an optional segmented calibrator routed by normalized raw score
* differentiable soft-QWK surrogate
* flexible decode helpers for threshold / expected-round / blend-round
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DecodeMode = Literal["threshold", "expected_round", "blend_round"]


@dataclass(frozen=True)
class DecodeSpec:
    mode: DecodeMode
    blend_alpha: float = 1.0
    max_delta: float | None = None
    max_delta_frac: float = 1.0


@dataclass
class CalibratorConfig:
    input_dim: int
    score_min: int
    score_max: int
    hidden_dim: int = 512
    dropout: float = 0.1
    lambda_qwk: float = 0.25
    lambda_reg: float = 1.0e-5

    @property
    def num_classes(self) -> int:
        return self.score_max - self.score_min + 1


@dataclass
class MMDConfig:
    enable: bool = False
    feature_space: str = "remb"
    scope: str = "prompt_wise"
    band_mode: str = "adjacent_only"
    sample_mode: str = "boundary_only"
    warmup_epochs: int = 5
    kernel: str = "rbf"
    sigma_mode: str = "median_heuristic"
    sigma: float = 1.0
    margin: float = 1.0
    lambda_sep: float = 0.0
    min_samples_per_band: int = 4
    boundary_mode: str = "uncertainty"
    uncertainty_threshold: float = 0.0
    raw_boundary_epsilon: float = 1.0
    num_bands: int = 5
    project_dim: int = 0
    project_dropout: float = 0.0
    eps: float = 1.0e-8


def raw_scores_from_feature(
    z: torch.Tensor,
    score_min: int,
    score_max: int,
) -> torch.Tensor:
    raw_norm = z[:, 0]
    score_span = float(score_max - score_min)
    return raw_norm * score_span + float(score_min)


def decode_coral(cum_probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Hard decode ordered bands from cumulative probabilities."""
    return (cum_probs > threshold).sum(dim=1).long()


def score_to_band_ids(
    scores: torch.Tensor,
    score_min: int,
    score_max: int,
    num_bands: int,
) -> torch.Tensor:
    if num_bands < 2:
        raise ValueError("num_bands must be >= 2")
    span = max(1.0, float(score_max - score_min))
    frac = (scores.float() - float(score_min)) / span
    frac = frac.clamp(min=0.0, max=1.0)
    return torch.clamp((frac * float(num_bands)).long(), max=num_bands - 1)


def raw_distance_to_nearest_boundary(
    raw_scores: torch.Tensor,
    score_min: int,
    score_max: int,
    num_bands: int,
) -> torch.Tensor:
    if num_bands < 2:
        return torch.full_like(raw_scores.float(), float("inf"))
    boundaries = torch.linspace(
        float(score_min),
        float(score_max),
        steps=num_bands + 1,
        device=raw_scores.device,
        dtype=raw_scores.float().dtype,
    )[1:-1]
    if boundaries.numel() == 0:
        return torch.full_like(raw_scores.float(), float("inf"))
    dists = torch.abs(raw_scores.float().unsqueeze(1) - boundaries.unsqueeze(0))
    return dists.min(dim=1).values


def _finite_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got {tuple(x.shape)}")
    mask = torch.isfinite(x).all(dim=1)
    return x[mask]


def _median_heuristic_sigma(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    joined = torch.cat([x, y], dim=0)
    if joined.shape[0] < 2:
        return torch.tensor(1.0, device=joined.device, dtype=joined.dtype)
    d2 = torch.cdist(joined, joined, p=2).pow(2)
    mask = torch.triu(torch.ones_like(d2, dtype=torch.bool), diagonal=1)
    vals = d2[mask]
    vals = vals[torch.isfinite(vals) & (vals > eps)]
    if vals.numel() == 0:
        return torch.tensor(1.0, device=joined.device, dtype=joined.dtype)
    sigma2 = vals.median().clamp_min(eps)
    return torch.sqrt(sigma2)


def _rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    sigma2 = torch.clamp(sigma.pow(2), min=eps)
    d2 = torch.cdist(x, y, p=2).pow(2)
    d2 = torch.where(torch.isfinite(d2), d2, torch.zeros_like(d2))
    return torch.exp(-d2 / (2.0 * sigma2))


def mmd_rbf_squared(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sigma_mode: str = "median_heuristic",
    sigma: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    x = _finite_rows(x.float())
    y = _finite_rows(y.float())
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.tensor(0.0, device=x.device if x.numel() else y.device)

    if sigma_mode == "fixed":
        sigma_t = torch.tensor(max(float(sigma), eps), device=x.device, dtype=x.dtype)
    else:
        sigma_t = _median_heuristic_sigma(x, y, eps)

    k_xx = _rbf_kernel(x, x, sigma_t, eps).mean()
    k_yy = _rbf_kernel(y, y, sigma_t, eps).mean()
    k_xy = _rbf_kernel(x, y, sigma_t, eps).mean()
    mmd2 = k_xx + k_yy - 2.0 * k_xy
    return torch.clamp(torch.where(torch.isfinite(mmd2), mmd2, torch.zeros_like(mmd2)), min=0.0)


def boundary_aware_mmd_separation_loss(
    *,
    features: torch.Tensor,
    prompt_ids: torch.Tensor,
    y_true: torch.Tensor,
    y_raw: torch.Tensor,
    uncertainty: torch.Tensor,
    score_min: int,
    score_max: int,
    config: MMDConfig,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    device = features.device
    zero = torch.tensor(0.0, device=device)
    empty_stats: Dict[str, object] = {
        "loss_sep": 0.0,
        "num_prompts_used_for_mmd": 0,
        "num_adjacent_pairs_used": 0,
        "num_samples_used_for_mmd": 0,
        "avg_mmd_value": 0.0,
        "avg_margin_gap": 0.0,
        "per_prompt_mmd": {},
    }
    if (not config.enable) or config.lambda_sep <= 0.0:
        return zero, empty_stats
    if config.feature_space != "remb":
        raise ValueError(f"Unsupported MMD feature_space: {config.feature_space}")
    if config.scope != "prompt_wise":
        raise ValueError(f"Unsupported MMD scope: {config.scope}")
    if config.band_mode != "adjacent_only":
        raise ValueError(f"Unsupported MMD band_mode: {config.band_mode}")
    if config.kernel != "rbf":
        raise ValueError(f"Unsupported MMD kernel: {config.kernel}")

    total_losses: list[torch.Tensor] = []
    pair_mmd_values: list[float] = []
    pair_margin_gaps: list[float] = []
    num_samples_used = 0
    prompts_used = 0
    per_prompt_mmd: Dict[str, Dict[str, float | int]] = {}
    band_ids = score_to_band_ids(y_true, score_min, score_max, config.num_bands)
    boundary_dist = raw_distance_to_nearest_boundary(
        y_raw,
        score_min,
        score_max,
        config.num_bands,
    )

    for prompt_id in prompt_ids.unique(sorted=True):
        p_mask = prompt_ids == prompt_id
        if not bool(p_mask.any()):
            continue
        p_feats = features[p_mask]
        p_bands = band_ids[p_mask]
        p_y_raw = y_raw[p_mask]
        p_unc = uncertainty[p_mask]
        p_boundary = boundary_dist[p_mask]
        p_losses: list[torch.Tensor] = []
        p_mmd_values: list[float] = []
        p_sample_count = 0

        for band_k in range(config.num_bands - 1):
            left = p_bands == band_k
            right = p_bands == (band_k + 1)
            if config.sample_mode == "boundary_only":
                if config.boundary_mode == "uncertainty":
                    left = left & (p_unc >= float(config.uncertainty_threshold))
                    right = right & (p_unc >= float(config.uncertainty_threshold))
                elif config.boundary_mode == "raw_boundary":
                    left = left & (p_boundary <= float(config.raw_boundary_epsilon))
                    right = right & (p_boundary <= float(config.raw_boundary_epsilon))
                else:
                    raise ValueError(
                        f"Unsupported MMD boundary_mode: {config.boundary_mode}"
                    )

            left_n = int(left.sum().item())
            right_n = int(right.sum().item())
            if left_n < int(config.min_samples_per_band) or right_n < int(config.min_samples_per_band):
                continue

            x = p_feats[left]
            y = p_feats[right]
            mmd2 = mmd_rbf_squared(
                x,
                y,
                sigma_mode=config.sigma_mode,
                sigma=config.sigma,
                eps=config.eps,
            )
            gap = torch.clamp(float(config.margin) - mmd2, min=0.0)
            p_losses.append(gap)
            total_losses.append(gap)
            p_mmd_values.append(float(mmd2.detach().cpu()))
            pair_mmd_values.append(float(mmd2.detach().cpu()))
            pair_margin_gaps.append(float(gap.detach().cpu()))
            p_sample_count += left_n + right_n
            num_samples_used += left_n + right_n

        if p_losses:
            prompts_used += 1
            per_prompt_mmd[str(int(prompt_id.item()))] = {
                "loss_sep": float(torch.stack(p_losses).mean().detach().cpu()),
                "avg_mmd_value": float(sum(p_mmd_values) / len(p_mmd_values)),
                "num_adjacent_pairs_used": len(p_losses),
                "num_samples_used_for_mmd": p_sample_count,
            }

    if total_losses:
        loss = torch.stack(total_losses).mean()
    else:
        loss = zero

    stats = {
        "loss_sep": float(loss.detach().cpu()),
        "num_prompts_used_for_mmd": prompts_used,
        "num_adjacent_pairs_used": len(total_losses),
        "num_samples_used_for_mmd": num_samples_used,
        "avg_mmd_value": float(sum(pair_mmd_values) / len(pair_mmd_values)) if pair_mmd_values else 0.0,
        "avg_margin_gap": float(sum(pair_margin_gaps) / len(pair_margin_gaps)) if pair_margin_gaps else 0.0,
        "per_prompt_mmd": per_prompt_mmd,
    }
    return loss, stats


def decode_scores(
    *,
    raw_score: torch.Tensor,
    expected_score: torch.Tensor,
    cum_probs: torch.Tensor,
    score_min: int,
    score_max: int,
    decode_mode: DecodeMode = "threshold",
    blend_alpha: float = 1.0,
    max_delta: float | None = None,
) -> torch.Tensor:
    if decode_mode == "threshold":
        pred = decode_coral(cum_probs).float() + float(score_min)
    elif decode_mode == "expected_round":
        pred = expected_score
    elif decode_mode == "blend_round":
        pred = raw_score + float(blend_alpha) * (expected_score - raw_score)
        if max_delta is not None:
            delta = (pred - raw_score).clamp(
                min=-float(max_delta),
                max=float(max_delta),
            )
            pred = raw_score + delta
    else:  # pragma: no cover - guarded by CLI choices
        raise ValueError(f"Unsupported decode_mode: {decode_mode}")
    pred = pred.round().clamp(min=score_min, max=score_max)
    return pred.long()


class CoralOrdinalCalibrator(nn.Module):
    """CORAL-style calibrator with ordered thresholds.

    ``z[:, 0]`` is assumed to be the normalized raw score in [0, 1]. The rest
    of the evidence vector is processed by the MLP trunk and then merged with a
    residual skip from ``y_raw``.
    """

    def __init__(self, config: CalibratorConfig) -> None:
        super().__init__()
        if config.num_classes < 2:
            raise ValueError("Need at least two score bands for ordinal calibration.")
        self.config = config
        trunk_in = max(1, config.input_dim - 1)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.base_head = nn.Linear(config.hidden_dim // 2, 1)
        self.raw_score_scale = nn.Parameter(torch.tensor(1.0))
        self.threshold_0 = nn.Parameter(torch.tensor(0.0))
        if config.num_classes > 2:
            self.threshold_gaps = nn.Parameter(torch.zeros(config.num_classes - 2))
        else:
            self.register_parameter("threshold_gaps", None)

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    def thresholds(self) -> torch.Tensor:
        if self.num_classes == 2:
            return self.threshold_0.unsqueeze(0)
        gaps = F.softplus(self.threshold_gaps)
        rest = self.threshold_0 + torch.cumsum(gaps, dim=0)
        return torch.cat([self.threshold_0.unsqueeze(0), rest], dim=0)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError(f"Expected z shape (batch, dim), got {tuple(z.shape)}")
        raw_score = z[:, :1]
        tail = z[:, 1:] if z.shape[1] > 1 else z[:, :1]
        trunk = self.trunk(tail)
        base = self.base_head(trunk)
        thresholds = self.thresholds().unsqueeze(0)
        cum_logits = base + self.raw_score_scale * raw_score - thresholds
        cum_probs = torch.sigmoid(cum_logits)
        band_probs = cumulative_to_band_probs(cum_probs)
        expected_band = band_probs @ torch.arange(
            self.num_classes, device=z.device, dtype=band_probs.dtype
        )
        expected_score = expected_band + self.config.score_min
        return {
            "base": base.squeeze(-1),
            "cum_logits": cum_logits,
            "cum_probs": cum_probs,
            "band_probs": band_probs,
            "expected_score": expected_score,
        }

    @torch.no_grad()
    def predict_scores(
        self,
        z: torch.Tensor,
        *,
        decode_mode: DecodeMode = "threshold",
        blend_alpha: float = 1.0,
        max_delta: float | None = None,
    ) -> torch.Tensor:
        out = self.forward(z)
        return decode_scores(
            raw_score=raw_scores_from_feature(z, self.config.score_min, self.config.score_max),
            expected_score=out["expected_score"],
            cum_probs=out["cum_probs"],
            score_min=self.config.score_min,
            score_max=self.config.score_max,
            decode_mode=decode_mode,
            blend_alpha=blend_alpha,
            max_delta=max_delta,
        )


def fit_segment_edges(
    raw_score_norm: torch.Tensor,
    num_segments: int,
    *,
    eps: float = 1.0e-6,
) -> list[float]:
    if num_segments <= 1 or raw_score_norm.numel() == 0:
        return []
    qs = torch.linspace(0.0, 1.0, num_segments + 1, dtype=torch.float32)[1:-1]
    quantiles = torch.quantile(raw_score_norm.detach().float().cpu(), qs).tolist()
    edges: list[float] = []
    for q in quantiles:
        qf = float(min(max(q, 0.0), 1.0))
        if not edges or abs(qf - edges[-1]) > eps:
            edges.append(qf)
    return edges


def route_segment_ids(
    raw_score_norm: torch.Tensor,
    segment_edges: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    raw_score_norm = raw_score_norm.contiguous()
    if isinstance(segment_edges, torch.Tensor):
        edges = segment_edges.to(device=raw_score_norm.device, dtype=raw_score_norm.dtype)
    else:
        if not segment_edges:
            return torch.zeros_like(raw_score_norm, dtype=torch.long)
        edges = torch.tensor(
            list(segment_edges),
            device=raw_score_norm.device,
            dtype=raw_score_norm.dtype,
        )
    if edges.numel() == 0:
        return torch.zeros_like(raw_score_norm, dtype=torch.long)
    return torch.bucketize(raw_score_norm, edges).long()


class SegmentedOrdinalCalibrator(nn.Module):
    """Piecewise calibrator routed by normalized raw score.

    Each segment owns its own CORAL calibrator. Routing is deterministic and
    based on quantile cut points learned from the train split's raw scores.
    """

    def __init__(self, config: CalibratorConfig, segment_edges: Sequence[float]) -> None:
        super().__init__()
        self.config = config
        self.segment_models = nn.ModuleList(
            [CoralOrdinalCalibrator(config) for _ in range(len(segment_edges) + 1)]
        )
        self.register_buffer(
            "segment_edges",
            torch.tensor(list(segment_edges), dtype=torch.float32),
            persistent=True,
        )

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    @property
    def num_segments(self) -> int:
        return len(self.segment_models)

    def thresholds(self) -> torch.Tensor:
        return torch.stack([m.thresholds() for m in self.segment_models], dim=0)

    def raw_score_scales(self) -> torch.Tensor:
        return torch.stack([m.raw_score_scale.detach() for m in self.segment_models], dim=0)

    def _segment_ids(self, z: torch.Tensor) -> torch.Tensor:
        return route_segment_ids(z[:, 0], self.segment_edges)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError(f"Expected z shape (batch, dim), got {tuple(z.shape)}")
        batch = z.shape[0]
        seg_ids = self._segment_ids(z)
        km1 = self.num_classes - 1
        base = torch.zeros(batch, device=z.device, dtype=z.dtype)
        cum_logits = torch.zeros(batch, km1, device=z.device, dtype=z.dtype)
        cum_probs = torch.zeros(batch, km1, device=z.device, dtype=z.dtype)
        band_probs = torch.zeros(batch, self.num_classes, device=z.device, dtype=z.dtype)
        expected_score = torch.zeros(batch, device=z.device, dtype=z.dtype)

        for seg_idx, submodel in enumerate(self.segment_models):
            mask = seg_ids == seg_idx
            if not bool(mask.any()):
                continue
            sub_out = submodel(z[mask])
            base[mask] = sub_out["base"]
            cum_logits[mask] = sub_out["cum_logits"]
            cum_probs[mask] = sub_out["cum_probs"]
            band_probs[mask] = sub_out["band_probs"]
            expected_score[mask] = sub_out["expected_score"]

        return {
            "base": base,
            "cum_logits": cum_logits,
            "cum_probs": cum_probs,
            "band_probs": band_probs,
            "expected_score": expected_score,
            "segment_ids": seg_ids,
        }

    @torch.no_grad()
    def predict_scores(
        self,
        z: torch.Tensor,
        *,
        decode_mode: DecodeMode = "threshold",
        blend_alpha: float = 1.0,
        max_delta: float | None = None,
    ) -> torch.Tensor:
        out = self.forward(z)
        return decode_scores(
            raw_score=raw_scores_from_feature(z, self.config.score_min, self.config.score_max),
            expected_score=out["expected_score"],
            cum_probs=out["cum_probs"],
            score_min=self.config.score_min,
            score_max=self.config.score_max,
            decode_mode=decode_mode,
            blend_alpha=blend_alpha,
            max_delta=max_delta,
        )


class MMDProjectionHead(nn.Module):
    """Small projector used only for the MMD branch."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Projection dimensions must be positive.")
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cumulative_to_band_probs(cum_probs: torch.Tensor) -> torch.Tensor:
    """Convert P(y > b_j) into class probabilities over ordered bands."""
    if cum_probs.ndim != 2:
        raise ValueError("cum_probs must have shape (batch, K-1)")
    batch, km1 = cum_probs.shape
    if km1 == 0:
        return torch.ones(batch, 1, device=cum_probs.device, dtype=cum_probs.dtype)
    out = []
    out.append(1.0 - cum_probs[:, 0])
    for j in range(1, km1):
        out.append(cum_probs[:, j - 1] - cum_probs[:, j])
    out.append(cum_probs[:, -1])
    probs = torch.stack(out, dim=1).clamp_min(1.0e-8)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return probs


def scores_to_band_indices(
    y_true: torch.Tensor,
    score_min: int,
    score_max: int,
) -> torch.Tensor:
    band = y_true.long() - int(score_min)
    if band.min().item() < 0 or band.max().item() > (score_max - score_min):
        raise ValueError("Ground-truth score outside configured range.")
    return band


def coral_targets(
    y_true: torch.Tensor,
    score_min: int,
    score_max: int,
) -> torch.Tensor:
    band_idx = scores_to_band_indices(y_true, score_min, score_max)
    thresholds = torch.arange(
        score_max - score_min, device=y_true.device, dtype=band_idx.dtype
    )
    return (band_idx.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def coral_ordinal_loss(
    cum_probs: torch.Tensor,
    y_true: torch.Tensor,
    score_min: int,
    score_max: int,
) -> torch.Tensor:
    targets = coral_targets(y_true, score_min, score_max)
    return F.binary_cross_entropy(cum_probs, targets)


def soft_qwk_score(
    band_probs: torch.Tensor,
    y_true: torch.Tensor,
    score_min: int,
    score_max: int,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    num_classes = score_max - score_min + 1
    band_true = scores_to_band_indices(y_true, score_min, score_max)
    one_hot = F.one_hot(band_true, num_classes=num_classes).float()
    observed = one_hot.transpose(0, 1) @ band_probs
    true_hist = one_hot.sum(dim=0)
    pred_hist = band_probs.sum(dim=0)
    expected = torch.outer(true_hist, pred_hist) / max(1, y_true.numel())
    grid = torch.arange(num_classes, device=band_probs.device, dtype=band_probs.dtype)
    weights = ((grid[:, None] - grid[None, :]) ** 2) / float((num_classes - 1) ** 2)
    num = (weights * observed).sum()
    den = (weights * expected).sum().clamp_min(eps)
    return 1.0 - (num / den)


def soft_qwk_loss(
    band_probs: torch.Tensor,
    y_true: torch.Tensor,
    score_min: int,
    score_max: int,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Differentiable surrogate loss for maximizing QWK."""
    score = soft_qwk_score(
        band_probs,
        y_true,
        score_min,
        score_max,
        eps=eps,
    )
    return 1.0 - score


def calibrator_loss(
    model: nn.Module,
    z: torch.Tensor,
    y_true: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    out = model(z)
    cfg = model.config
    loss_ord = coral_ordinal_loss(out["cum_probs"], y_true, cfg.score_min, cfg.score_max)
    loss_qwk = soft_qwk_loss(out["band_probs"], y_true, cfg.score_min, cfg.score_max)
    qwk_score = soft_qwk_score(out["band_probs"], y_true, cfg.score_min, cfg.score_max)
    reg = torch.tensor(0.0, device=z.device)
    for param in model.parameters():
        reg = reg + param.pow(2).sum()
    reg = cfg.lambda_reg * reg
    total = loss_ord + cfg.lambda_qwk * loss_qwk + reg
    return total, {
        "loss_ord": float(loss_ord.detach().cpu()),
        "loss_qwk": float(loss_qwk.detach().cpu()),
        "soft_qwk_score": float(qwk_score.detach().cpu()),
        "loss_reg": float(reg.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }


__all__ = [
    "MMDConfig",
    "MMDProjectionHead",
    "CalibratorConfig",
    "CoralOrdinalCalibrator",
    "DecodeMode",
    "DecodeSpec",
    "SegmentedOrdinalCalibrator",
    "boundary_aware_mmd_separation_loss",
    "calibrator_loss",
    "coral_ordinal_loss",
    "coral_targets",
    "cumulative_to_band_probs",
    "decode_coral",
    "decode_scores",
    "fit_segment_edges",
    "mmd_rbf_squared",
    "raw_scores_from_feature",
    "raw_distance_to_nearest_boundary",
    "route_segment_ids",
    "score_to_band_ids",
    "scores_to_band_indices",
    "soft_qwk_loss",
    "soft_qwk_score",
]
