from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_anchor_budget_experiment as runner  # noqa: E402
from scripts.bapr_repair import (  # noqa: E402
    DECOMPRESS_EXTREME_ANCHORS,
    REBALANCE_SCORE_BANDS,
    REPLACE_WORST_BAND_ANCHOR,
    boundary_improved_for_operator,
    compute_failure_profile,
    generate_repaired_children,
    guarded_select_repaired_bank,
    metrics_with_anchor_stats,
    rank_repair_operators,
    split_val_diag_sel,
)
from scripts.run_anchor_budget_experiment import AnchorRecord, is_run_complete  # noqa: E402


def _item(essay_id: int, score: int, text: str | None = None) -> dict:
    return {
        "essay_id": essay_id,
        "domain1_score": score,
        "essay_text": text or f"essay {essay_id} score {score} evidence organization language",
    }


def _anchor(essay_id: int, score: int, text: str | None = None) -> dict:
    return {
        "essay_id": essay_id,
        "gold_score": score,
        "prompt_id": 1,
        "token_length": len((text or "").split()) or 8,
        "source_split": "train",
        "selection_score": 1.0,
        "selection_reason": "toy",
        "essay_text": text or f"anchor {essay_id} score {score} evidence organization language",
    }


def _metrics(**overrides):
    base = {
        "qwk": 0.50,
        "mae": 1.00,
        "high_recall": 0.20,
        "high_tail_under_score_rate": 0.80,
        "max_recall": 0.00,
        "max_score_under_score_rate": 1.00,
        "range_coverage": 0.30,
        "score_compression_index": 0.50,
        "score_tv": 0.40,
        "worst_band_mae": 1.50,
        "anchor_band_coverage": 3,
        "anchor_unique_score_count": 3,
        "anchor_score_range_span": 10,
        "token_cost": 100,
    }
    base.update(overrides)
    return base


def _bank(candidate_id: str, operator: str, target_metrics: list[str]) -> dict:
    anchors = [_anchor(1, 2), _anchor(2, 7), _anchor(3, 12)]
    return {
        "candidate_id": candidate_id,
        "parent_id": "BAPR-A0",
        "operator": operator,
        "anchor_bank_id": candidate_id,
        "anchor_ids": [1, 2, 3],
        "anchor_scores": [2, 7, 12],
        "anchors": anchors,
        "target_boundary_metrics": target_metrics,
    }


def test_split_val_diag_sel_is_deterministic_and_disjoint():
    val = [_item(100 + i, score) for i, score in enumerate([2, 2, 4, 6, 7, 8, 10, 11, 12, 12])]
    diag1, sel1, meta1 = split_val_diag_sel(val, 2, 12, diag_ratio=0.5)
    diag2, sel2, meta2 = split_val_diag_sel(list(reversed(val)), 2, 12, diag_ratio=0.5)

    assert [x["essay_id"] for x in diag1] == [x["essay_id"] for x in diag2]
    assert [x["essay_id"] for x in sel1] == [x["essay_id"] for x in sel2]
    assert {x["essay_id"] for x in diag1}.isdisjoint({x["essay_id"] for x in sel1})
    assert meta1["val_diag_ids_hash"] == meta2["val_diag_ids_hash"]
    assert meta1["val_sel_ids_hash"] == meta2["val_sel_ids_hash"]


def test_empty_band_metrics_are_none_and_do_not_trigger_worst_band_repair():
    anchors = [_anchor(1, 12), _anchor(2, 11), _anchor(3, 10)]
    profile = compute_failure_profile([12, 11], [10, 10], anchors, 2, 12)
    ranked = rank_repair_operators(profile)

    assert profile["band_mae_low"] is None
    assert profile["band_mae_mid"] is None
    assert all(
        not (row["operator"] == REPLACE_WORST_BAND_ANCHOR and row.get("target_band") in {"low", "mid"})
        for row in ranked
    )


def test_generate_children_keeps_exact_k_unique_and_excludes_forbidden_ids():
    parent = [_anchor(1, 2), _anchor(2, 7), _anchor(3, 12)]
    train = [
        _item(1, 2),
        _item(2, 7),
        _item(3, 12),
        _item(4, 12, "excellent evidence top high"),
        _item(5, 12, "excellent evidence top high unique"),
        _item(6, 2, "weak low"),
    ]
    val_diag = [_item(101, 12, "excellent evidence top high")]
    ranked = [
        {
            "operator": DECOMPRESS_EXTREME_ANCHORS,
            "severity": 1.0,
            "target_band": "high",
            "target_boundary_metrics": ["high_recall"],
        }
    ]
    children, trace = generate_repaired_children(
        parent,
        train,
        val_diag,
        {"missing_anchor_bands": []},
        ranked,
        2,
        12,
        k=3,
        forbidden_ids={4, 101},
    )

    assert children
    assert trace
    child_ids = children[0]["anchor_ids"]
    assert len(child_ids) == 3
    assert len(set(child_ids)) == 3
    assert 4 not in child_ids


def test_guard_requires_operator_related_boundary_metric_improvement():
    parent = {
        "anchor_bank_id": "parent",
        "anchor_ids": [1, 2, 3],
        "anchor_scores": [2, 7, 12],
    }
    unrelated_child = _bank("child_unrelated", DECOMPRESS_EXTREME_ANCHORS, ["high_recall"])
    accepted_child = _bank("child_high", DECOMPRESS_EXTREME_ANCHORS, ["high_recall"])
    parent_metrics = _metrics()
    child_metrics = [
        _metrics(qwk=0.51, score_tv=0.20, high_recall=0.20),
        _metrics(qwk=0.51, high_recall=0.40),
    ]

    guard = guarded_select_repaired_bank(parent_metrics, child_metrics, parent, [unrelated_child, accepted_child])
    rows = {row["candidate_id"]: row for row in guard["selection_rows"]}

    assert rows["child_unrelated"]["accepted_by_guard"] is False
    assert "target_boundary_metric_not_improved" in rows["child_unrelated"]["guard_reject_reasons"]
    assert rows["child_high"]["accepted_by_guard"] is True
    assert guard["selected_anchor_bank"]["candidate_id"] == "child_high"


def test_guard_rejects_qwk_collapse_and_falls_back_to_parent_when_all_children_rejected():
    parent = {
        "anchor_bank_id": "parent",
        "anchor_ids": [1, 2, 3],
        "anchor_scores": [2, 7, 12],
    }
    child = _bank("child_bad", DECOMPRESS_EXTREME_ANCHORS, ["high_recall"])
    guard = guarded_select_repaired_bank(
        _metrics(qwk=0.50),
        [_metrics(qwk=0.10, high_recall=0.50)],
        parent,
        [child],
        {"bapr": {"guard": {"qwk_drop_tolerance": 0.02}}},
    )

    assert guard["selected_anchor_bank"] is parent
    assert guard["selected_reason"] == "parent_fallback_all_children_rejected"
    assert guard["selection_rows"][0]["candidate_id"] == "BAPR-A0"
    assert guard["selection_rows"][0]["selected_as_final"] is True


def test_score_compression_index_improvement_uses_closeness_to_one():
    assert boundary_improved_for_operator(
        _metrics(score_compression_index=0.40),
        _metrics(score_compression_index=0.85),
        ["score_compression_index"],
    )
    assert not boundary_improved_for_operator(
        _metrics(score_compression_index=0.85),
        _metrics(score_compression_index=1.40),
        ["score_compression_index"],
    )


def test_guard_tie_break_is_deterministic_by_candidate_id():
    parent = {
        "anchor_bank_id": "parent",
        "anchor_ids": [1, 2, 3],
        "anchor_scores": [2, 7, 12],
    }
    child_2 = _bank("child_2", DECOMPRESS_EXTREME_ANCHORS, ["high_recall"])
    child_1 = _bank("child_1", DECOMPRESS_EXTREME_ANCHORS, ["high_recall"])
    guard = guarded_select_repaired_bank(
        _metrics(qwk=0.50),
        [_metrics(qwk=0.55, mae=0.90, high_recall=0.60), _metrics(qwk=0.55, mae=0.90, high_recall=0.60)],
        parent,
        [child_2, child_1],
    )

    assert guard["selected_anchor_bank"]["candidate_id"] == "child_1"


def test_bapr_run_complete_requires_bapr_specific_outputs(tmp_path):
    for name in runner.REQUIRED_RUN_FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    assert is_run_complete(tmp_path, "retrieval_k_anchor")
    assert not is_run_complete(tmp_path, "bapr_repair_k_anchor")

    for name in [
        "bapr_failure_profile.json",
        "bapr_repair_candidates.jsonl",
        "bapr_guarded_selection.csv",
        "bapr_parent_anchor_bank.json",
        "bapr_parent_metrics.json",
        "bapr_final_anchor_bank.json",
        "bapr_repair_trace.jsonl",
    ]:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    assert is_run_complete(tmp_path, "bapr_repair_k_anchor")


def test_run_bapr_one_uses_v_diag_not_v_sel_for_parent_a0(monkeypatch, tmp_path):
    train = [_item(i, score) for i, score in enumerate([2, 3, 7, 8, 11, 12], start=1)]
    val = [_item(100 + i, score) for i, score in enumerate([2, 2, 7, 7, 12, 12])]
    test = [_item(200 + i, score) for i, score in enumerate([2, 7, 12])]
    expected_diag, expected_sel, _ = split_val_diag_sel(val, 2, 12, 0.5)
    captured = {}

    def fake_parent_builder(train_rows, val_rows, k, score_min, score_max, config, instruction, backend, out_dir):
        captured["val_ids"] = [int(x["essay_id"]) for x in val_rows]
        anchors = [
            AnchorRecord(
                essay_id=int(x["essay_id"]),
                gold_score=int(x["domain1_score"]),
                prompt_id=1,
                token_length=8,
                source_split="train",
                selection_score=1.0,
                selection_reason="fake_a0",
                essay_text=x["essay_text"],
            )
            for x in train_rows[:k]
        ]
        return anchors, [{"essay_id": a.essay_id, "gold_score": a.gold_score} for a in anchors]

    monkeypatch.setattr(runner, "retrieval_grounded_stratified_rep_anchors_v21", fake_parent_builder)
    summary = runner.run_bapr_one(
        method="bapr_repair_k_anchor",
        k=3,
        config={
            "data": {"score_min": 2, "score_max": 12, "essay_set": 1},
            "bapr": {"val_diag_ratio": 0.5, "max_children": 0},
        },
        fold=0,
        seed=42,
        train=train,
        val=val,
        test=test,
        instruction="Rubric",
        backend=None,
        root_out=tmp_path,
        split_hashes={},
    )

    assert captured["val_ids"] == [int(x["essay_id"]) for x in expected_diag]
    assert set(captured["val_ids"]).isdisjoint({int(x["essay_id"]) for x in expected_sel})
    out_dir = Path(summary["exp_dir"])
    assert (out_dir / "bapr_parent_anchor_bank.json").exists()
    assert (out_dir / "bapr_parent_metrics.json").exists()
    assert (out_dir / "bapr_final_anchor_bank.json").exists()


def test_bapr_config_loads():
    import yaml

    with open("configs/anchor_budget_bapr_v1.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["anchor_budget"]["methods"] == ["bapr_repair_k_anchor"]
    assert cfg["pace"]["final_pace_calibrated"] is False
