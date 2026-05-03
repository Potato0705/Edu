"""APE-lite baseline for WISE-PACE.

Generates several instruction candidates in one pass, evaluates them with the
same local scoring model and fixed stratified anchors, and saves the best.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_evidence_mutation.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-candidates", type=int, default=4)
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
    prompt = f"""Generate {args.n_candidates} distinct operational scoring instructions for automated essay scoring.

Official criteria:
{official}

Each instruction should be concise but specific, include score-boundary guidance,
and be directly usable by an LLM grader. Return a numbered list only."""
    text = generate_text(prompt, call_type="ape_lite_generate")
    candidates = parse_numbered_candidates(text)[: args.n_candidates]
    if not candidates:
        raise SystemExit("APE-lite did not produce usable instruction candidates.")

    results = []
    for idx, instruction in enumerate(candidates):
        qwk = evaluate_instruction(
            instruction=instruction,
            anchors=anchors,
            val_set=val_set,
            config=config,
        )
        print(f"candidate={idx} val_qwk={qwk:.4f}")
        results.append({"candidate_id": idx, "val_qwk": qwk, "instruction": instruction})

    best = max(results, key=lambda x: x["val_qwk"])
    path = save_result(
        args.output_dir,
        "ape_lite",
        {
            "config": args.config,
            "fold": args.fold,
            "anchor_ids": [x["essay_id"] for x in anchors],
            "anchor_scores": [x["domain1_score"] for x in anchors],
            "best": best,
            "candidates": results,
        },
    )
    print(f"best_val_qwk={best['val_qwk']:.4f}")
    print(f"saved={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
