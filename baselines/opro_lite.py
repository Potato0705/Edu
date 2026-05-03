"""OPRO-lite baseline for WISE-PACE.

This is a small, budget-controlled prompt optimization baseline: it keeps fixed
stratified anchors and asks the local LLM to propose new scoring instructions
from the best validation scores observed so far.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.common import (
    evaluate_instruction,
    fixed_anchors,
    generate_text,
    init_local_runtime,
    load_config,
    load_debug_split,
    parse_numbered_candidates,
    save_result,
)


def propose(history: list[dict], official: str, n_candidates: int, iteration: int) -> list[str]:
    ranked = sorted(history, key=lambda x: x["val_qwk"], reverse=True)[:5]
    history_text = "\n\n".join(
        f"Score={item['val_qwk']:.4f}\nInstruction:\n{item['instruction']}"
        for item in ranked
    )
    prompt = f"""You are optimizing an essay-scoring instruction.

Official criteria:
{official}

Previous candidates and validation QWK:
{history_text}

Propose {n_candidates} improved instructions for iteration {iteration}.
Prioritize sharper score-boundary rules, observable writing traits, and fewer ambiguous criteria.
Return a numbered list only."""
    return parse_numbered_candidates(generate_text(prompt, call_type="opro_lite_propose"))[:n_candidates]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_evidence_mutation.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/baselines"))
    args = parser.parse_args()

    config = load_config(args.config)
    init_local_runtime(config)
    train_set, val_set, _test_set = load_debug_split(config, args.fold)
    anchors = fixed_anchors(train_set, val_set, config)
    official = config.get("induction", {}).get(
        "official_criteria",
        "Evaluate the essay based on content, organization, and language use.",
    )

    history = []
    seed_instruction = official
    seed_qwk = evaluate_instruction(
        instruction=seed_instruction,
        anchors=anchors,
        val_set=val_set,
        config=config,
    )
    history.append({"iteration": 0, "candidate_id": 0, "val_qwk": seed_qwk, "instruction": seed_instruction})
    print(f"iteration=0 candidate=0 val_qwk={seed_qwk:.4f}")

    for iteration in range(1, args.iterations + 1):
        candidates = propose(history, official, args.n_candidates, iteration)
        for idx, instruction in enumerate(candidates):
            qwk = evaluate_instruction(
                instruction=instruction,
                anchors=anchors,
                val_set=val_set,
                config=config,
            )
            print(f"iteration={iteration} candidate={idx} val_qwk={qwk:.4f}")
            history.append(
                {
                    "iteration": iteration,
                    "candidate_id": idx,
                    "val_qwk": qwk,
                    "instruction": instruction,
                }
            )

    best = max(history, key=lambda x: x["val_qwk"])
    path = save_result(
        args.output_dir,
        "opro_lite",
        {
            "config": args.config,
            "fold": args.fold,
            "iterations": args.iterations,
            "n_candidates": args.n_candidates,
            "anchor_ids": [x["essay_id"] for x in anchors],
            "anchor_scores": [x["domain1_score"] for x in anchors],
            "best": best,
            "history": history,
        },
    )
    print(f"best_val_qwk={best['val_qwk']:.4f}")
    print(f"saved={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
