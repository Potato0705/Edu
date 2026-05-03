"""Oracle-style raw scoring ceiling diagnostic.

This is not a WISE-PACE training run. It evaluates whether the target scorer
LLM can achieve reasonable raw direct AES performance when given a strong,
fixed scoring instruction and boundary-aware gold anchors. If this ceiling is
low, the bottleneck is likely scorer-model capability or scoring format. If it
is high but evolution is low, the bottleneck is the I/E search process.
"""

from __future__ import annotations

import argparse
import json
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
from wise_aes import EvolutionOptimizer, PromptIndividual


def build_oracle_instruction(config: dict) -> str:
    score_min = config["data"]["score_min"]
    score_max = config["data"]["score_max"]
    official = config.get("induction", {}).get(
        "official_criteria",
        "Evaluate the essay based on content, organization, and language use.",
    )
    return wise_aes._prepend_score_range_contract(
        f"""You are a strict, consistent automated essay scorer.

Official scoring criteria:
{official}

Use the full {score_min}-{score_max} integer scale. Treat the provided anchors
as global score-scale references. Compare the target essay against the low,
boundary, and high anchors before assigning a final score.

Decision rules:
- Assign low scores only when development, organization, or language control is clearly weak.
- Use middle and boundary scores for mixed essays; distinguish adjacent scores by concrete evidence, not by general impressions.
- Assign high scores only when ideas are well developed, organization is controlled, and language use supports meaning.
- The final score must match the reasoning and must be one integer in [{score_min}, {score_max}].""",
        config,
    )


def evaluate_protocol(
    *,
    label: str,
    instruction: str,
    anchors: list[dict],
    train_set: list[dict],
    val_set: list[dict],
    test_set: list[dict],
    config: dict,
) -> dict:
    optimizer = EvolutionOptimizer(train_set, val_set, config)
    protocol = PromptIndividual(instruction, anchors, config=config)
    val_qwk = protocol.evaluate(
        val_set,
        optimizer.vector_store,
        enable_rerank=config.get("rag", {}).get("use_rerank_train", False),
    )
    val_metrics = optimizer._score_prediction_metrics(
        [x["domain1_score"] for x in val_set],
        protocol.last_pred_scores,
    )
    test_qwk = protocol.evaluate(
        test_set,
        optimizer.vector_store,
        enable_rerank=config.get("rag", {}).get("use_rerank_test", False),
    )
    test_metrics = optimizer._score_prediction_metrics(
        [x["domain1_score"] for x in test_set],
        protocol.last_pred_scores,
    )
    return {
        "label": label,
        "raw_val_qwk": float(val_qwk),
        "raw_val_metrics": val_metrics,
        "raw_test_qwk": float(test_qwk),
        "raw_test_metrics": test_metrics,
        "instruction": instruction,
        "anchor_ids": [x["essay_id"] for x in anchors],
        "anchor_scores": [x["domain1_score"] for x in anchors],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_evidence_mutation.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model-path", default=None, help="Optional scorer model override for one-model diagnosis.")
    parser.add_argument("--instruction-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/baselines"))
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model_path:
        config.setdefault("pace", {})["model_path"] = args.model_path
        config.setdefault("model", {})["name"] = args.model_path
    init_local_runtime(config)
    train_set, val_set, test_set = load_debug_split(config, args.fold)
    anchors = fixed_anchors(train_set, val_set, config)
    if args.instruction_file:
        instruction = wise_aes._prepend_score_range_contract(
            args.instruction_file.read_text(encoding="utf-8"),
            config,
        )
        instruction_source = str(args.instruction_file)
    else:
        instruction = build_oracle_instruction(config)
        instruction_source = "built_in_oracle_template"

    result = evaluate_protocol(
        label="oracle_raw_ceiling",
        instruction=instruction,
        anchors=anchors,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        config=config,
    )
    payload = {
        "baseline": "oracle_raw_ceiling",
        "purpose": (
            "Diagnose scorer LLM raw direct scoring ceiling with a strong fixed instruction "
            "and boundary-aware anchors. Calibrated scores are not used."
        ),
        "config": args.config,
        "fold": args.fold,
        "model_path": config.get("pace", {}).get("model_path"),
        "instruction_source": instruction_source,
        "result": result,
        "usage": {
            "prompt_tokens": wise_aes.EXP_MANAGER.total_prompt_tokens if wise_aes.EXP_MANAGER else 0,
            "completion_tokens": wise_aes.EXP_MANAGER.total_completion_tokens if wise_aes.EXP_MANAGER else 0,
            "tokens_total": wise_aes.EXP_MANAGER.total_tokens if wise_aes.EXP_MANAGER else 0,
        },
    }
    path = save_result(args.output_dir, "oracle_ceiling", payload)
    print(json.dumps({
        "raw_val_qwk": result["raw_val_qwk"],
        "raw_test_qwk": result["raw_test_qwk"],
        "raw_test_mae": result["raw_test_metrics"].get("mae"),
        "anchor_scores": result["anchor_scores"],
        "saved": str(path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
