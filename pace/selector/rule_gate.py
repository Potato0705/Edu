"""Rule gate for narrowing the PARS recipe candidate set."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from pace.selector.recipe_library import Recipe


@dataclass
class RuleGateConfig:
    score_span_fine: float = 5.0
    score_span_large: float = 20.0
    off_by1_high: float = 0.25
    off_by2plus_high: float = 0.12
    mean_band_distance_high: float = 0.40
    adjacent_overlap_high: float = 1.0
    min_adjacent_pair_count: int = 4
    raw_mode_share_high: float = 0.60
    raw_std_frac_low: float = 0.18


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _remove_mmd_candidates(candidates: List[str], recipes: Dict[str, Recipe]) -> List[str]:
    kept = [rid for rid in candidates if not recipes[rid].mmd_enable]
    return kept or candidates


def _available(candidates: List[str], recipes: Dict[str, Recipe]) -> List[str]:
    return [rid for rid in candidates if rid in recipes]


def apply_rule_gate(
    features: Dict[str, object],
    recipes: Dict[str, Recipe],
    config: RuleGateConfig,
) -> Dict[str, object]:
    score_span = float(features.get("score_span", 0.0))
    off_by1 = float(features.get("off_by_1_share", 0.0))
    off_by2plus = float(features.get("off_by_2plus_share", 0.0))
    mean_band_dist = float(features.get("mean_band_distance_errors", 0.0))
    raw_mode_share = float(features.get("raw_mode_share", 0.0))
    raw_std_frac = float(features.get("raw_std_frac", 1.0))

    fine = (
        score_span <= float(config.score_span_fine)
        and off_by1 >= float(config.off_by1_high)
        and off_by2plus < float(config.off_by2plus_high)
    )
    wide = (
        score_span >= float(config.score_span_large)
        or off_by2plus >= float(config.off_by2plus_high)
        or mean_band_dist >= float(config.mean_band_distance_high)
    )
    raw_collapse = (
        raw_mode_share >= float(config.raw_mode_share_high)
        or raw_std_frac <= float(config.raw_std_frac_low)
    )
    wide_domain = score_span >= float(config.score_span_large)

    reasons: List[str] = []
    if not wide_domain:
        if fine:
            prompt_type = "fine_grained"
            reasons.append("non-wide: fine-grained score range")
        else:
            prompt_type = "non_wide_default"
            reasons.append("non-wide: restrict to threshold recipes")
        candidates = _available(["R0", "R1", "R2"], recipes)
        if wide:
            reasons.append("non-wide: severe errors do not enable wide recipes")
    elif fine:
        prompt_type = "fine_grained"
        candidates = _available(["R0", "R1", "R2"], recipes)
        reasons.append("fine: small span + high off-by-1 + low off-by-2plus")
    elif wide:
        if raw_collapse:
            prompt_type = "wide_range_raw_collapse"
            candidates = _available(["R3", "R5", "R7", "R6"], recipes)
            if len(candidates) > 3:
                candidates = candidates[:3]
            reasons.append("wide: raw prediction collapse, allow larger correction cap")
        else:
            prompt_type = "wide_range"
            candidates = _available(["R3", "R4"], recipes)
            reasons.append("wide: large span or high severe boundary error")
    else:
        prompt_type = "middle_default"
        candidates = _available(["R1", "R2"], recipes)
        reasons.append("default: middle-range diagnostics")

    if not candidates:
        candidates = _available(["R1"], recipes) or list(recipes.keys())[:1]

    overlap = features.get("adjacent_overlap_score", float("nan"))
    pair_count = int(features.get("min_adjacent_pair_count", 0) or 0)
    enough_overlap = (
        _is_finite(overlap)
        and float(overlap) >= float(config.adjacent_overlap_high)
        and pair_count >= int(config.min_adjacent_pair_count)
    )
    if not enough_overlap:
        candidates = _remove_mmd_candidates(candidates, recipes)
        reasons.append("mmd_gate: removed MMD recipes")
    else:
        reasons.append("mmd_gate: kept MMD recipes")

    return {
        "prompt_type": prompt_type,
        "candidate_recipes": candidates,
        "candidate_recipe_str": ",".join(candidates),
        "mmd_gate_kept": bool(enough_overlap),
        "raw_collapse_gate": bool(raw_collapse),
        "rule_gate_reasons": "; ".join(reasons),
    }
