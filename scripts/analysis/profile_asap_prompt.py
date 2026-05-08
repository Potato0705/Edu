"""Profile an ASAP prompt without calling any LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def profile_type_from_span(score_span: int | float) -> str:
    if score_span <= 6:
        return "narrow"
    if score_span <= 12:
        return "medium"
    if score_span <= 30:
        return "wide"
    return "ultrawide"


def strategy_for_profile(profile_type: str) -> Dict[str, object]:
    if profile_type == "narrow":
        return {
            "recommended_anchor_count": 3,
            "recommended_mutation_types": ["boundary_clarification_mutation"],
            "recommended_selection_metrics": ["qwk", "mae_tiebreak"],
            "recommended_cost_settings": {"top_k_pace": 1, "max_new_tokens_scoring": 160},
        }
    if profile_type == "medium":
        return {
            "recommended_anchor_count": 4,
            "recommended_mutation_types": ["high_tail_instruction_mutation", "boundary_clarification_mutation"],
            "recommended_selection_metrics": ["qwk", "high_recall", "distribution_tv"],
            "recommended_cost_settings": {"top_k_pace": 1, "max_new_tokens_scoring": 192},
        }
    if profile_type == "wide":
        return {
            "recommended_anchor_count": 5,
            "recommended_mutation_types": ["score_distribution_mutation", "boundary_clarification_mutation"],
            "recommended_selection_metrics": ["qwk", "mae", "distribution_tv"],
            "recommended_cost_settings": {"top_k_pace": 1, "max_new_tokens_scoring": 192},
        }
    return {
        "recommended_anchor_count": "5-7",
        "recommended_mutation_types": ["score_distribution_mutation", "boundary_clarification_mutation"],
        "recommended_selection_metrics": ["coarse_qwk", "mae", "round_number_compression"],
        "recommended_cost_settings": {"top_k_pace": 1, "max_new_tokens_scoring": 192, "staged_validation": True},
    }


def profile_prompt_dataframe(df: pd.DataFrame, prompt: int) -> Dict[str, object]:
    sub = df[df["essay_set"].astype(int) == int(prompt)].copy()
    if sub.empty:
        raise ValueError(f"Prompt {prompt} not found")
    scores = sub["domain1_score"].astype(int).to_numpy()
    essays = sub["essay"].astype(str)
    lengths = essays.map(lambda x: len(x.split())).to_numpy(dtype=float)
    score_min = int(scores.min())
    score_max = int(scores.max())
    score_span = int(score_max - score_min)
    profile_type = profile_type_from_span(score_span)
    high_threshold = int(np.ceil(score_min + 0.75 * max(1, score_span)))
    low_max = int(np.floor(score_min + max(1, score_span) / 3.0))
    high_min = int(np.ceil(score_min + 2.0 * max(1, score_span) / 3.0))

    score_hist = {
        str(score): int((scores == score).sum())
        for score in range(score_min, score_max + 1)
    }
    high_mask = scores >= high_threshold
    risk_flags: List[str] = []
    if int((scores == score_max).sum()) < 5:
        risk_flags.append("few_max_score_examples")
    if float(high_mask.mean()) < 0.10:
        risk_flags.append("rare_high_scores")
    if len(set(scores.tolist())) <= 3:
        risk_flags.append("low_score_cardinality")
    if score_span > 30:
        risk_flags.append("ultrawide_score_range")

    strategy = strategy_for_profile(profile_type)
    payload: Dict[str, object] = {
        "prompt": int(prompt),
        "n_examples": int(len(sub)),
        "score_min": score_min,
        "score_max": score_max,
        "score_span": score_span,
        "unique_score_count": int(len(set(scores.tolist()))),
        "score_histogram": score_hist,
        "score_quantiles": {
            "q10": float(np.quantile(scores, 0.10)),
            "q25": float(np.quantile(scores, 0.25)),
            "q50": float(np.quantile(scores, 0.50)),
            "q75": float(np.quantile(scores, 0.75)),
            "q90": float(np.quantile(scores, 0.90)),
        },
        "max_score_count": int((scores == score_max).sum()),
        "high_score_threshold": high_threshold,
        "high_score_count": int(high_mask.sum()),
        "high_score_ratio": float(high_mask.mean()),
        "essay_length": {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "q25": float(np.quantile(lengths, 0.25)),
            "q50": float(np.quantile(lengths, 0.50)),
            "q75": float(np.quantile(lengths, 0.75)),
        },
        "high_score_essay_length": {
            "mean": float(np.mean(lengths[high_mask])) if bool(high_mask.any()) else 0.0,
            "std": float(np.std(lengths[high_mask])) if bool(high_mask.any()) else 0.0,
        },
        "low_anchor_pool_count": int((scores <= low_max).sum()),
        "mid_anchor_pool_count": int(((scores > low_max) & (scores < high_min)).sum()),
        "high_anchor_pool_count": int((scores >= high_min).sum()),
        "profile_type": profile_type,
        "risk_flags": risk_flags,
    }
    payload.update(strategy)
    return payload


def write_profile(profile: Dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# ASAP Prompt {profile['prompt']} Profile",
        "",
        f"- score range: {profile['score_min']}-{profile['score_max']}",
        f"- profile type: {profile['profile_type']}",
        f"- examples: {profile['n_examples']}",
        f"- high score ratio: {profile['high_score_ratio']:.3f}",
        f"- risk flags: {', '.join(profile['risk_flags']) if profile['risk_flags'] else 'none'}",
        f"- recommended anchors: {profile['recommended_anchor_count']}",
        f"- recommended mutation types: {', '.join(profile['recommended_mutation_types'])}",
    ]
    (out_dir / "prompt_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=int, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    data_path = args.data_path
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        data_path = data_path or cfg.get("data", {}).get("asap_path")
    if not data_path:
        raise SystemExit("--data-path or --config with data.asap_path is required")
    df = pd.read_csv(data_path, sep="\t", encoding="latin1")
    profile = profile_prompt_dataframe(df, args.prompt)
    profile["fold"] = int(args.fold)
    out_dir = Path(args.out_dir or f"outputs/prompt_profiles/p{args.prompt}_fold{args.fold}")
    write_profile(profile, out_dir)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print(f"Profile written to {out_dir}")


if __name__ == "__main__":
    main()
