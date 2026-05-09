"""
Test A: PaceFitnessEvaluator 单元测试
在不运行完整进化循环的情况下，验证 pace/pace_fitness.py 端到端流程。

运行方式：
  cd wise-aes
  python test_pace_fitness_quick.py
"""

import sys
import os
import time

import pandas as pd
import torch

# 切换到 wise-aes 目录（保证相对 import 正确）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "data/raw/training_set_rel3.tsv"
ESSAY_SET = 1
SCORE_MIN, SCORE_MAX = 2, 12
LOAD_IN_4BIT = os.environ.get("WISE_PACE_LOAD_IN_4BIT", "0") == "1"


def load_sample_essays(n=10):
    df = pd.read_csv(DATA_PATH, sep="\t", encoding="latin-1")
    df = df[df["essay_set"] == ESSAY_SET].head(n)
    return [
        {
            "essay_id": int(row["essay_id"]),
            "essay_text": str(row["essay"]),
            "domain1_score": int(row["domain1_score"]),
        }
        for _, row in df.iterrows()
    ]


def separator(title):
    print(f"\n{'='*20} {title} {'='*20}")


def check(condition, msg):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        raise AssertionError(f"CHECK FAILED: {msg}")


def main():
    separator("Test A: PaceFitnessEvaluator Unit Test")

    # --- 1. 加载数据 ---
    separator("Step 1: Load sample essays")
    essays = load_sample_essays(n=10)
    print(f"  Loaded {len(essays)} essays (set={ESSAY_SET})")
    print(f"  Scores: {[e['domain1_score'] for e in essays]}")
    check(len(essays) >= 5, "至少加载到 5 篇 essay")

    # --- 2. 初始化 LocalLlamaBackend ---
    quant_label = "4-bit" if LOAD_IN_4BIT else "bfloat16"
    separator(f"Step 2: Load LocalLlamaBackend ({quant_label})")
    from pace.llm_backend import LocalLlamaBackend

    # 构造最小 config
    config = {
        "llm": {"temperature_scoring": 0.0},
        "data": {"score_min": SCORE_MIN, "score_max": SCORE_MAX},
        "rag": {},
    }
    t0 = time.time()
    backend = LocalLlamaBackend(
        config=config,
        model_path=MODEL_PATH,
        dtype="bfloat16",
        load_in_4bit=LOAD_IN_4BIT,
    )
    print(f"  Backend loaded in {time.time()-t0:.1f}s. hidden_dim={backend.hidden_dim}")
    check(backend.hidden_dim > 0, f"hidden_dim={backend.hidden_dim}")

    # --- 3. 初始化 PaceFitnessEvaluator ---
    separator("Step 3: Init PaceFitnessEvaluator")
    from pace.pace_fitness import PaceFitnessConfig, PaceFitnessEvaluator

    pace_config = PaceFitnessConfig(
        top_k_pace=1,
        calib_split_ratio=0.5,
        alpha=0.70,
        beta=0.30,
        hidden_dim=64,
        epochs=5,           # 测试时少跑几 epoch
        lr=1e-3,
        use_compact_evidence=True,
    )
    evaluator = PaceFitnessEvaluator(
        local_backend=backend,
        config=pace_config,
        score_min=SCORE_MIN,
        score_max=SCORE_MAX,
    )
    print("  PaceFitnessEvaluator created OK")

    # --- 4. score_essays：检查 hidden shape ---
    separator("Step 4: score_essays (5 essays)")
    dummy_instruction = "Grade the essay from 2 to 12. Output JSON: {\"final_score\": <int>}"
    dummy_exemplars = essays[:3]   # 用前 3 篇当 anchor exemplar
    test_essays = essays[3:8]

    t0 = time.time()
    results = evaluator.score_essays(test_essays, dummy_instruction, dummy_exemplars)
    elapsed = time.time() - t0
    print(f"  Scored {len(results)} essays in {elapsed:.1f}s")

    for i, r in enumerate(results):
        check(r.hidden is not None, f"essay {i}: hidden is not None")
        check(r.hidden.shape == (backend.hidden_dim,), f"essay {i}: hidden.shape == ({backend.hidden_dim},)")
        check(SCORE_MIN <= r.y_raw <= SCORE_MAX, f"essay {i}: y_raw={r.y_raw} in [{SCORE_MIN},{SCORE_MAX}]")
        print(f"    essay_id={r.essay_id} y_raw={r.y_raw} hidden.shape={tuple(r.hidden.shape)}")

    # --- 5. compute_anchor_hiddens ---
    separator("Step 5: compute_anchor_hiddens (3 anchors)")
    anchor_essays = essays[:3]
    t0 = time.time()
    anchor_hiddens = evaluator.compute_anchor_hiddens(anchor_essays, dummy_instruction)
    elapsed = time.time() - t0
    print(f"  Anchor hiddens computed in {elapsed:.1f}s: shape={tuple(anchor_hiddens.shape)}")
    check(anchor_hiddens.shape == (3, backend.hidden_dim),
          f"anchor_hiddens.shape == (3, {backend.hidden_dim})")

    # --- 6. _build_evidence_bundle → z.shape=(35,) ---
    separator("Step 6: _build_evidence_bundle (compact, 35-dim)")
    z = evaluator._build_evidence_bundle(results[0], test_essays[0]["essay_text"], anchor_hiddens)
    print(f"  z.shape = {tuple(z.shape)}")
    # 1 + 3 + 3 + 11 + 11 + 6 = 35
    expected_dim = 1 + anchor_hiddens.shape[0] + anchor_hiddens.shape[0] + 11 + 11 + 6
    check(z.shape == (expected_dim,), f"z.shape == ({expected_dim},)")
    check(torch.isfinite(z).all(), "z 中无 NaN/Inf")

    # --- 7. _train_calibrator ---
    separator("Step 7: _train_calibrator")
    # 构造 calib 数据（5 个样本）
    all_results = evaluator.score_essays(essays[:5], dummy_instruction, dummy_exemplars)
    z_list = []
    y_list = []
    for item, result in zip(essays[:5], all_results):
        if result.hidden is not None:
            z_i = evaluator._build_evidence_bundle(result, item["essay_text"], anchor_hiddens)
            z_list.append(z_i)
            y_list.append(item["domain1_score"])

    z_calib = torch.stack(z_list)
    y_calib = torch.tensor(y_list, dtype=torch.long)
    print(f"  Training calibrator on z={z_calib.shape}, y={y_calib.tolist()}")

    calibrator = evaluator._train_calibrator(z_calib, y_calib)
    print("  Calibrator trained OK")

    # predict_scores
    calibrator.eval()
    with torch.no_grad():
        preds = calibrator.predict_scores(z_calib.to(evaluator._device), decode_mode="threshold")
    preds_list = preds.cpu().tolist()
    print(f"  Predictions: {preds_list}")
    for p in preds_list:
        check(SCORE_MIN <= p <= SCORE_MAX, f"pred={p} in [{SCORE_MIN},{SCORE_MAX}]")

    # --- 8. compute_pace_fitness 端到端 ---
    separator("Step 8: compute_pace_fitness end-to-end")
    # 构造一个 mock protocol 对象
    class MockProtocol:
        instruction_text = dummy_instruction
        static_exemplars = anchor_essays

    calib_items = essays[:5]
    fitness_items = essays[5:]
    print(f"  calib={len(calib_items)} fitness={len(fitness_items)}")

    t0 = time.time()
    result = evaluator.compute_pace_fitness(MockProtocol(), calib_items, fitness_items)
    elapsed = time.time() - t0
    print(f"  compute_pace_fitness returned in {elapsed:.1f}s")
    print(f"  Result: {result}")

    required_keys = ["pace_qwk", "raw_qwk", "combined_fitness",
                     "anchor_inference_sec", "calib_inference_sec",
                     "fitness_inference_sec", "calibrator_train_sec"]
    for k in required_keys:
        check(k in result, f"result contains key '{k}'")

    if not result.get("_fallback"):
        check(-1.0 <= result["pace_qwk"] <= 1.0, f"pace_qwk={result['pace_qwk']:.4f} in [-1,1]")
        check(-1.0 <= result["raw_qwk"] <= 1.0, f"raw_qwk={result['raw_qwk']:.4f} in [-1,1]")
        check(torch.isfinite(torch.tensor(result["combined_fitness"])),
              f"combined_fitness={result['combined_fitness']:.4f} is finite")
    else:
        print("  [INFO] PACE fell back to raw QWK (expected with tiny dataset)")

    separator("ALL TESTS PASSED")
    print("\nSummary:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  hidden_dim: {backend.hidden_dim}")
    print(f"  z_dim: {expected_dim}")
    if not result.get("_fallback"):
        print(f"  pace_qwk:  {result['pace_qwk']:.4f}")
        print(f"  raw_qwk:   {result['raw_qwk']:.4f}")
        print(f"  combined:  {result['combined_fitness']:.4f}")
    print(f"  Total time: {time.time() - t0:.0f}s (step 8 only)")


if __name__ == "__main__":
    main()
