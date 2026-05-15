from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_anchor_budget_experiment as runner  # noqa: E402
from scripts.bapr_repair import (  # noqa: E402
    DECOMPRESS_EXTREME_ANCHORS,
    REBALANCE_SCORE_BANDS,
    REPLACE_WORST_BAND_ANCHOR,
    band_for as bapr_band_for,
    boundary_improved_for_operator,
    compute_failure_profile,
    generate_repaired_children,
    guarded_select_repaired_bank,
    metrics_with_anchor_stats,
    rank_repair_operators,
    split_val_diag_sel,
)
from scripts.anchor_influence import (  # noqa: E402
    estimate_proxy_influence,
    generate_influence_repair_children,
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
        "anchor_score_range_coverage": 1.0,
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


def test_rebalance_accepts_anchor_coverage_improvement():
    parent = {
        "anchor_bank_id": "parent",
        "anchor_ids": [1, 2, 3],
        "anchor_scores": [4, 5, 6],
    }
    child = _bank(
        "child_rebalance",
        REBALANCE_SCORE_BANDS,
        ["anchor_band_coverage", "anchor_unique_score_count", "anchor_score_range_span", "anchor_score_range_coverage"],
    )
    guard = guarded_select_repaired_bank(
        _metrics(qwk=0.50, mae=1.00, anchor_band_coverage=2, anchor_unique_score_count=3, anchor_score_range_span=4),
        [_metrics(qwk=0.50, mae=1.00, anchor_band_coverage=3, anchor_unique_score_count=3, anchor_score_range_span=4)],
        parent,
        [child],
    )

    row = {r["candidate_id"]: r for r in guard["selection_rows"]}["child_rebalance"]
    assert row["accepted_by_guard"] is True
    assert row["target_boundary_metric_improved"] is True
    assert guard["selected_anchor_bank"]["candidate_id"] == "child_rebalance"


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


def test_guard_uses_named_parent_candidate_id_for_fallback():
    parent = {
        "candidate_id": "retrieval_diag_parent",
        "anchor_bank_id": "parent",
        "anchor_ids": [1, 2, 3],
        "anchor_scores": [2, 7, 12],
    }
    guard = guarded_select_repaired_bank(_metrics(), [], parent, [])

    assert guard["selected_anchor_bank"] is parent
    assert guard["selection_rows"][0]["candidate_id"] == "retrieval_diag_parent"
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


def test_anchor_redundancy_metrics_improve_when_lower():
    assert boundary_improved_for_operator(
        _metrics(anchor_redundancy_mean=0.60, anchor_redundancy_max=0.80),
        _metrics(anchor_redundancy_mean=0.30, anchor_redundancy_max=0.70),
        ["anchor_redundancy_mean"],
    )
    assert boundary_improved_for_operator(
        _metrics(anchor_redundancy_mean=0.60, anchor_redundancy_max=0.80),
        _metrics(anchor_redundancy_mean=0.70, anchor_redundancy_max=0.50),
        ["anchor_redundancy_max"],
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


def test_bapr_band_for_matches_runner_score_band_label():
    ranges = [(2, 12)]
    for path in ["configs/anchor_budget_phase2_p2.yaml", "configs/anchor_budget_phase2_p7.yaml"]:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ranges.append((int(cfg["data"]["score_min"]), int(cfg["data"]["score_max"])))
    for score_min, score_max in ranges:
        for score in range(score_min, score_max + 1):
            assert bapr_band_for(score, score_min, score_max) == runner.band_for(score, score_min, score_max)


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


def test_bapr_si_run_complete_requires_stability_and_influence_outputs(tmp_path):
    for name in runner.REQUIRED_RUN_FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
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

    assert not is_run_complete(tmp_path, "bapr_si_k_anchor")
    for name in [
        "anchor_stability_scores.csv",
        "stability_trace.jsonl",
        "anchor_influence_scores.csv",
        "anchor_influence_trace.jsonl",
        "anchor_loo_influence_scores.csv",
        "anchor_influence_child_alignment.csv",
        "bapr_si_repair_trace.jsonl",
    ]:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert is_run_complete(tmp_path, "bapr_si_k_anchor")


def test_proxy_influence_identifies_non_stable_harmful_anchor():
    parent = [
        _anchor(1, 2, "low weak vague"),
        _anchor(2, 7, "middle repeated repeated"),
        _anchor(3, 8, "middle repeated repeated"),
        _anchor(4, 12, "high excellent"),
    ]
    stability_rows = [
        {"essay_id": 1, "stability_score": 0.8, "selection_frequency": 1.0},
        {"essay_id": 2, "stability_score": 0.1, "selection_frequency": 0.2},
        {"essay_id": 3, "stability_score": 0.1, "selection_frequency": 0.2},
        {"essay_id": 4, "stability_score": 0.7, "selection_frequency": 0.9},
    ]
    rows, trace = estimate_proxy_influence(
        parent,
        [],
        [],
        {"score_compression_index": 0.45, "range_coverage": 0.20, "high_tail_under_score_rate": 0.0},
        stability_rows,
        score_min=2,
        score_max=12,
    )

    assert rows
    assert trace
    assert rows[0]["anchor_failure_type"] in {"harmful_compression_anchor", "redundant_low_influence_anchor"}
    assert rows[0]["negative_influence_score"] > 0


def test_proxy_influence_does_not_create_boundary_confuser_without_band_error():
    parent = [
        _anchor(1, 2, "alpha banana cedar"),
        _anchor(2, 7, "delta ember frost"),
        _anchor(3, 12, "gamma harbor ivory"),
    ]
    stability_rows = [
        {"essay_id": 1, "stability_score": 0.8, "selection_frequency": 1.0},
        {"essay_id": 2, "stability_score": 0.8, "selection_frequency": 1.0},
        {"essay_id": 3, "stability_score": 0.8, "selection_frequency": 1.0},
    ]
    rows, _ = estimate_proxy_influence(
        parent,
        [],
        [],
        {
            "worst_band": "mid",
            "band_mae_mid": 0.0,
            "score_compression_index": 1.0,
            "range_coverage": 1.0,
            "high_tail_under_score_rate": 0.0,
            "max_score_under_score_rate": 0.0,
        },
        stability_rows,
        score_min=2,
        score_max=12,
    )

    by_id = {row["anchor_id"]: row for row in rows}
    assert by_id[2]["anchor_failure_type"] == "useful_stabilizing_anchor"


def test_influence_repair_children_replace_negative_anchor_with_stable_candidate():
    parent = [_anchor(1, 2), _anchor(2, 7), _anchor(3, 12)]
    train = [_item(1, 2), _item(2, 7), _item(3, 12), _item(4, 7, "stable mid candidate"), _item(5, 12)]
    stability_rows = [
        {"essay_id": 4, "gold_score": 7, "band": "mid", "stability_score": 0.95, "selection_frequency": 1.0},
        {"essay_id": 5, "gold_score": 12, "band": "high", "stability_score": 0.50, "selection_frequency": 0.5},
    ]
    influence_rows = [
        {
            "anchor_id": 2,
            "anchor_failure_type": "boundary_confuser",
            "negative_influence_score": 0.8,
            "target_band": "mid",
            "target_boundary_metrics": ["worst_band_mae"],
            "expected_repair_metric": "worst_band_mae",
        }
    ]

    children, trace = generate_influence_repair_children(
        parent,
        train,
        [],
        stability_rows,
        influence_rows,
        score_min=2,
        score_max=12,
        k=3,
        forbidden_ids={100, 101},
        max_children=1,
    )

    assert children
    assert trace
    assert 2 not in children[0]["anchor_ids"]
    assert 4 in children[0]["anchor_ids"]
    assert children[0]["removed_anchor_failure_type"] == "boundary_confuser"
    assert children[0]["added_anchor_stability"] == pytest.approx(0.95)


def test_bapr_v1_config_v21_params_are_used_by_selector(tmp_path):
    with open("configs/anchor_budget_bapr_v1.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    items = []
    essay_id = 1
    for score in range(2, 13):
        for j in range(3):
            items.append(
                _item(
                    essay_id,
                    score,
                    f"score {score} sample {j} evidence organization language " + ("excellent " * max(0, score - 7)),
                )
            )
            essay_id += 1
    val = [
        _item(1001, 2, "weak limited low"),
        _item(1002, 7, "middle evidence organized"),
        _item(1003, 12, "excellent high evidence sophisticated"),
    ]

    anchors, trace = runner.retrieval_grounded_stratified_rep_anchors_v21(
        items, val, 9, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path
    )

    assert len(anchors) == 9
    assert trace
    assert all(row["retrieval_weight"] == 0.8 for row in trace)
    assert all(row["representation_weight"] == 0.2 for row in trace)
    assert all(row["fallback_margin"] == 0.08 for row in trace)
    assert all(row["max_rep_replacements"] == 3 for row in trace)


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


def test_run_bapr_one_passes_val_and_test_ids_as_forbidden(monkeypatch, tmp_path):
    train = [_item(i, score) for i, score in enumerate([2, 3, 7, 8, 11, 12], start=1)]
    val = [_item(100 + i, score) for i, score in enumerate([2, 2, 7, 7, 12, 12])]
    test = [_item(200 + i, score) for i, score in enumerate([2, 7, 12])]
    expected_diag, expected_sel, _ = split_val_diag_sel(val, 2, 12, 0.5)
    captured = {}

    def fake_parent_builder(train_rows, val_rows, k, score_min, score_max, config, instruction, backend, out_dir):
        anchors = [
            runner.AnchorRecord(
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

    def fake_generate(parent_anchors, train_pool, val_diag, failure_profile, ranked_operators, score_min, score_max, k, forbidden_ids=(), max_children=3):
        captured["forbidden_ids"] = set(int(x) for x in forbidden_ids)
        return [], []

    monkeypatch.setattr(runner, "retrieval_grounded_stratified_rep_anchors_v21", fake_parent_builder)
    monkeypatch.setattr(runner, "generate_repaired_children", fake_generate)
    runner.run_bapr_one(
        method="bapr_repair_k_anchor",
        k=3,
        config={
            "data": {"score_min": 2, "score_max": 12, "essay_set": 1},
            "bapr": {"val_diag_ratio": 0.5, "max_children": 1},
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

    expected_forbidden = (
        {int(x["essay_id"]) for x in expected_diag}
        | {int(x["essay_id"]) for x in expected_sel}
        | {int(x["essay_id"]) for x in test}
    )
    assert captured["forbidden_ids"] == expected_forbidden
    assert captured["forbidden_ids"].isdisjoint({int(x["essay_id"]) for x in train})


def test_run_bapr_one_retrieval_parent_uses_v_diag_only(monkeypatch, tmp_path):
    train = [_item(i, score) for i, score in enumerate([2, 3, 4, 7, 8, 9, 10, 11, 12], start=1)]
    val = [_item(100 + i, score) for i, score in enumerate([2, 2, 7, 7, 12, 12])]
    test = [_item(200 + i, score) for i, score in enumerate([2, 7, 12])]
    expected_diag, expected_sel, _ = split_val_diag_sel(val, 2, 12, 0.5)
    captured = {}
    real_retrieval = runner.retrieval_anchors

    def wrapped_retrieval(train_rows, val_rows, k, score_min, score_max):
        captured["val_ids"] = [int(x["essay_id"]) for x in val_rows]
        return real_retrieval(train_rows, val_rows, k, score_min, score_max)

    monkeypatch.setattr(runner, "retrieval_anchors", wrapped_retrieval)
    summary = runner.run_bapr_one(
        method="bapr_repair_k_anchor",
        k=3,
        config={
            "data": {"score_min": 2, "score_max": 12, "essay_set": 1},
            "bapr": {"parent_init_method": "retrieval_k_anchor", "val_diag_ratio": 0.5, "max_children": 0},
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
    run_dir = Path(summary["exp_dir"])
    parent_bank = yaml.safe_load((run_dir / "bapr_parent_anchor_bank.json").read_text(encoding="utf-8"))
    final_bank = yaml.safe_load((run_dir / "bapr_final_anchor_bank.json").read_text(encoding="utf-8"))
    assert parent_bank["method"] == "retrieval_diag_parent"
    assert parent_bank["candidate_id"] == "retrieval_diag_parent"
    assert parent_bank["bapr_parent_init_method"] == "retrieval_k_anchor"
    assert final_bank["method"] == "BAPR-retrieval-A*"
    assert final_bank["bapr_parent_init_method"] == "retrieval_k_anchor"


def test_bapr_fake_scoring_smoke_writes_required_outputs(monkeypatch, tmp_path):
    train = [_item(i, score) for i, score in enumerate([2, 3, 4, 7, 8, 9, 10, 11, 12], start=1)]
    val = [_item(100 + i, score) for i, score in enumerate([2, 2, 7, 7, 12, 12])]
    test = [_item(200 + i, score) for i, score in enumerate([2, 7, 12])]

    def fake_parent_builder(train_rows, val_rows, k, score_min, score_max, config, instruction, backend, out_dir):
        anchors = [
            runner.AnchorRecord(
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

    run_dir = Path(summary["exp_dir"])
    for name in [
        "bapr_val_split.json",
        "bapr_parent_anchor_bank.json",
        "bapr_parent_metrics.json",
        "bapr_failure_profile.json",
        "bapr_repair_candidates.jsonl",
        "bapr_guarded_selection.csv",
        "bapr_final_anchor_bank.json",
        "bapr_repair_trace.jsonl",
    ]:
        assert (run_dir / name).exists()
    with open(run_dir / "predictions.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert all("fake_scoring" in row["raw_text"] for row in rows)


def test_bapr_si_fake_scoring_smoke_writes_stability_influence_outputs(tmp_path):
    train = [_item(i, score) for i, score in enumerate([2, 3, 4, 6, 7, 8, 10, 11, 12], start=1)]
    val = [_item(100 + i, score) for i, score in enumerate([2, 2, 7, 7, 12, 12])]
    test = [_item(200 + i, score) for i, score in enumerate([2, 7, 12])]
    summary = runner.run_bapr_one(
        method="bapr_si_k_anchor",
        k=3,
        config={
            "data": {"score_min": 2, "score_max": 12, "essay_set": 1},
            "anchor_budget": {"stability_retrieval": {"n_bootstrap": 3, "per_band_top_n": 3}},
            "bapr": {
                "repair_mode": "stability_influence",
                "parent_init_method": "stability_retrieval_k_anchor",
                "val_diag_ratio": 0.5,
                "max_children": 1,
                "influence": {
                    "loo_attribution_enabled": True,
                    "loo_max_anchors": 3,
                    "loo_max_items": 3,
                },
            },
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

    run_dir = Path(summary["exp_dir"])
    for name in [
        "bapr_val_split.json",
        "bapr_parent_anchor_bank.json",
        "bapr_parent_metrics.json",
        "bapr_failure_profile.json",
        "bapr_repair_candidates.jsonl",
        "bapr_guarded_selection.csv",
        "bapr_final_anchor_bank.json",
        "bapr_repair_trace.jsonl",
        "anchor_stability_scores.csv",
        "stability_trace.jsonl",
        "anchor_influence_scores.csv",
        "anchor_influence_trace.jsonl",
        "anchor_loo_influence_scores.csv",
        "anchor_influence_child_alignment.csv",
        "bapr_si_repair_trace.jsonl",
    ]:
        assert (run_dir / name).exists()
    with open(run_dir / "anchor_loo_influence_scores.csv", encoding="utf-8", newline="") as f:
        loo_rows = list(csv.DictReader(f))
    assert loo_rows
    assert {"delta_qwk_without_anchor", "loo_harm_score", "proxy_failure_type"} <= set(loo_rows[0])
    final_bank = yaml.safe_load((run_dir / "bapr_final_anchor_bank.json").read_text(encoding="utf-8"))
    assert final_bank["bapr_repair_mode"] == "stability_influence"


def test_bapr_si_reuses_parent_stability_artifacts(monkeypatch, tmp_path):
    train = [_item(i, score) for i, score in enumerate([2, 3, 4, 6, 7, 8, 10, 11, 12], start=1)]
    val = [_item(100 + i, score) for i, score in enumerate([2, 2, 7, 7, 12, 12])]
    test = [_item(200 + i, score) for i, score in enumerate([2, 7, 12])]
    calls = {"n": 0}

    def fake_estimate(train_pool, val_diag, *, k, score_min, score_max, **kwargs):
        calls["n"] += 1
        rows = []
        for idx, item in enumerate(train_pool):
            score = int(item["domain1_score"])
            rows.append(
                {
                    "essay_id": int(item["essay_id"]),
                    "gold_score": score,
                    "band": runner.band_for(score, score_min, score_max),
                    "selection_frequency": 1.0,
                    "mean_rank": idx + 1,
                    "rank_variance": 0.0,
                    "mean_retrieval_score": float(len(train_pool) - idx),
                    "redundancy_score": 0.0,
                    "stability_score": float(len(train_pool) - idx),
                    "selected_count": 1,
                    "bootstrap_count": 1,
                    "token_length": 8,
                }
            )
        return rows, [{"bootstrap_index": 0, "essay_id": int(train_pool[0]["essay_id"])}]

    monkeypatch.setattr(runner, "estimate_anchor_stability", fake_estimate)
    runner.run_bapr_one(
        method="bapr_si_k_anchor",
        k=3,
        config={
            "data": {"score_min": 2, "score_max": 12, "essay_set": 1},
            "anchor_budget": {"stability_retrieval": {"n_bootstrap": 3, "per_band_top_n": 3}},
            "bapr": {
                "repair_mode": "stability_influence",
                "parent_init_method": "stability_retrieval_k_anchor",
                "val_diag_ratio": 0.5,
                "max_children": 0,
                "influence": {"loo_attribution_enabled": False},
            },
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

    assert calls["n"] == 1


def test_bapr_config_loads():
    with open("configs/anchor_budget_bapr_v1.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["anchor_budget"]["methods"] == ["bapr_repair_k_anchor"]
    assert cfg["pace"]["final_pace_calibrated"] is False
    assert "retrieval_weight" not in cfg["anchor_budget"]["representation"]
    assert cfg["anchor_budget"]["retrieval_grounded_rep_v21"]["retrieval_weight"] == 0.8
    assert cfg["anchor_budget"]["retrieval_grounded_rep_v21"]["representation_weight"] == 0.2
    assert cfg["bapr"]["parent_init_method"] == "retrieval_grounded_stratified_rep_k_anchor_v21"
    with open("configs/anchor_budget_bapr_v1_retrieval_parent.yaml", encoding="utf-8") as f:
        retrieval_cfg = yaml.safe_load(f)
    assert retrieval_cfg["bapr"]["parent_init_method"] == "retrieval_k_anchor"
