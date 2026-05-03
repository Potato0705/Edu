"""Calibrator-only diagnostic baseline for WISE-PACE.

This baseline intentionally performs no protocol evolution. It uses a fixed
official instruction plus fixed anchors, then trains the PACE calibrator on the
validation calibration split. Its purpose is to separate "calibrator can repair
scores" from "WISE-PACE found a better I/E protocol for direct raw scoring".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wise_aes
from baselines.common import (
    fixed_anchors,
    init_local_runtime,
    load_config,
    load_debug_split,
    save_result,
)
from pace.pace_fitness import PaceFitnessConfig, PaceFitnessEvaluator
from wise_aes import EvolutionOptimizer, PromptIndividual


def split_calib_eval(val_set: list[dict], ratio: float) -> tuple[list[dict], list[dict]]:
    desired = int(len(val_set) * ratio)
    n_calib = max(4, desired)
    if len(val_set) >= 8:
        n_calib = min(n_calib, len(val_set) - 4)
    else:
        n_calib = max(0, len(val_set) // 2)
    return val_set[:n_calib], val_set[n_calib:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_evidence_mutation.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/baselines"))
    args = parser.parse_args()

    config = load_config(args.config)
    backend = init_local_runtime(config)
    train_set, val_set, test_set = load_debug_split(config, args.fold)
    anchors = fixed_anchors(train_set, val_set, config)
    official = config.get("induction", {}).get(
        "official_criteria",
        "Evaluate the essay based on content, organization, and language use.",
    )
    instruction = wise_aes._prepend_score_range_contract(official, config)
    protocol = PromptIndividual(instruction, anchors, config=config)

    pace_cfg = config.get("pace", {})
    calib_items, val_eval_items = split_calib_eval(
        val_set, float(pace_cfg.get("calib_split_ratio", 0.5))
    )
    if not calib_items or not val_eval_items:
        raise SystemExit("Validation split is too small for calibrator-only baseline.")

    optimizer = EvolutionOptimizer(train_set, val_set, config)
    raw_val_qwk = protocol.evaluate(
        val_eval_items,
        optimizer.vector_store,
        enable_rerank=config.get("rag", {}).get("use_rerank_train", False),
    )
    raw_test_qwk = protocol.evaluate(
        test_set,
        optimizer.vector_store,
        enable_rerank=config.get("rag", {}).get("use_rerank_test", False),
    )

    evaluator = PaceFitnessEvaluator(
        local_backend=backend,
        config=PaceFitnessConfig.from_config(config),
        score_min=config["data"]["score_min"],
        score_max=config["data"]["score_max"],
    )
    val_probe = evaluator.score_with_pace_calibrator(protocol, calib_items, val_eval_items)
    test_probe = evaluator.score_with_pace_calibrator(protocol, calib_items, test_set)

    payload = {
        "baseline": "calibrator_only_fixed_protocol",
        "purpose": (
            "Diagnostic only: measures how much a post-hoc PACE calibrator can repair "
            "a fixed protocol. This is not counted as WISE-PACE protocol-evolution performance."
        ),
        "config": args.config,
        "fold": args.fold,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_calib": len(calib_items),
        "n_val_eval": len(val_eval_items),
        "anchor_ids": [x["essay_id"] for x in anchors],
        "anchor_scores": [x["domain1_score"] for x in anchors],
        "instruction": instruction,
        "raw_val_qwk_direct": float(raw_val_qwk),
        "raw_test_qwk_direct": float(raw_test_qwk),
        "calibrated_val_probe": val_probe,
        "calibrated_test_probe": test_probe,
        "summary": {
            "raw_val_qwk_same_pass": val_probe.get("raw_qwk"),
            "calibrated_val_qwk": val_probe.get("pace_qwk"),
            "raw_test_qwk_same_pass": test_probe.get("raw_qwk"),
            "calibrated_test_qwk": test_probe.get("pace_qwk"),
            "test_calibration_gain": (
                float(test_probe.get("pace_qwk", 0.0) or 0.0)
                - float(test_probe.get("raw_qwk", 0.0) or 0.0)
            ),
            "pace_qwk_counts_as_main_result": False,
        },
        "usage": {
            "prompt_tokens": wise_aes.EXP_MANAGER.total_prompt_tokens if wise_aes.EXP_MANAGER else 0,
            "completion_tokens": wise_aes.EXP_MANAGER.total_completion_tokens if wise_aes.EXP_MANAGER else 0,
            "tokens_total": wise_aes.EXP_MANAGER.total_tokens if wise_aes.EXP_MANAGER else 0,
            "backend_usage": backend.usage_snapshot(),
        },
    }
    path = save_result(args.output_dir, "calibrator_only", payload)
    print(f"raw_val_qwk_direct={raw_val_qwk:.4f}")
    print(f"raw_test_qwk_direct={raw_test_qwk:.4f}")
    print(f"calibrated_val_qwk={float(val_probe.get('pace_qwk', 0.0) or 0.0):.4f}")
    print(f"calibrated_test_qwk={float(test_probe.get('pace_qwk', 0.0) or 0.0):.4f}")
    print("calibrated scores are diagnostic only; raw_test_qwk_direct is the direct-scoring baseline.")
    print(f"saved={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
