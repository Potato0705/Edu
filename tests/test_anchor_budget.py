from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_anchor_budget_experiment as runner  # noqa: E402
from scripts.run_anchor_budget_experiment import (  # noqa: E402
    deterministic_score_covered,
    coverage_first_sisa_anchors,
    load_asap_split,
    phase1_decision,
    representation_guided_anchors,
    retrieval_anchors,
    retrieval_grounded_no_rep_anchors,
    retrieval_grounded_stratified_rep_anchors,
    retrieval_grounded_stratified_rep_anchors_v21,
    score_boundary_metrics,
    sisa_anchors,
    stability_retrieval_anchors,
    stratified_rep_guided_anchors,
    stratified_anchors,
)
from scripts.anchor_stability import (  # noqa: E402
    estimate_anchor_stability,
    select_coverage_first_sisa_anchor_rows,
    select_sisa_anchor_rows,
    select_stable_anchor_rows,
    target_score_ladder,
)


def _toy_items():
    rows = []
    eid = 1
    for score in range(2, 13):
        for j in range(3):
            text = (
                f"essay score {score} sample {j} "
                + ("excellent evidence organization language " * score)
                + ("weak vague " * max(0, 12 - score))
            )
            rows.append({"essay_id": eid, "essay_text": text, "domain1_score": score})
            eid += 1
    return rows


def test_score_boundary_metrics_include_compression_and_recall():
    metrics = score_boundary_metrics([2, 8, 10, 12], [2, 8, 8, 10], 2, 12)
    assert metrics["high_recall"] == 0.0
    assert metrics["max_recall"] == 0.0
    assert 0.0 < metrics["score_compression_index"] < 1.0
    assert metrics["range_coverage"] == pytest.approx(3 / 11)
    assert metrics["high_tail_under_score_rate"] == 1.0


def test_static_and_stratified_anchor_selection_are_score_range_generic():
    items = _toy_items()
    static = deterministic_score_covered(items, 6, 2, 12, seed=42, method="static_k_anchor")
    stratified = stratified_anchors(items, 6, 2, 12, seed=42)
    assert len(static) == 6
    assert len(stratified) == 6
    assert len({a.gold_score for a in static}) >= 5
    assert {"low", "mid", "high"} <= {
        "low" if a.gold_score <= 5 else "high" if a.gold_score >= 9 else "mid"
        for a in stratified
    }


def test_retrieval_anchors_use_score_slot_coverage_for_budgeted_selection():
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    anchors = retrieval_anchors(items, val, 9, 2, 12)
    scores = [a.gold_score for a in anchors]

    assert len(anchors) == 9
    assert any(score <= 3 for score in scores)
    assert any(score in {6, 7} for score in scores)
    assert any(score >= 11 for score in scores)


def test_representation_guided_tfidf_changes_anchor_choice(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "anchor_budget": {
            "representation": {
                "mode": "tfidf",
                "max_candidates_per_score": 3,
                "token_cost_penalty": 0.0,
            }
        },
    }
    anchors, trace = representation_guided_anchors(
        items,
        val,
        4,
        2,
        12,
        cfg,
        "Rubric",
        backend=None,
        out_dir=tmp_path,
    )
    assert len(anchors) == 4
    assert len(trace) == 4
    assert all("selection_parts" in row for row in trace)
    assert all(a.selection_reason.startswith("representation_guided:tfidf") for a in anchors)


def test_representation_guided_local_hidden_uses_same_val_sample_for_high_indices(tmp_path):
    class FakeBackend:
        def encode_scoring_context(self, **kwargs):
            text = kwargs.get("essay_text", "")
            score = float(kwargs.get("known_score") or (12 if "excellent" in text else 2))
            return torch.tensor([score, len(text) % 7, 1.0], dtype=torch.float32)

    items = _toy_items()
    # More than max_val_representations, with high-score items after the cutoff.
    val = [
        {"essay_id": 2000 + i, "essay_text": f"validation low {i}", "domain1_score": 2}
        for i in range(6)
    ] + [
        {"essay_id": 3000 + i, "essay_text": f"excellent high validation {i}", "domain1_score": 12}
        for i in range(6)
    ]
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "anchor_budget": {
            "representation": {
                "mode": "local_hidden",
                "max_candidates_per_score": 2,
                "max_val_representations": 4,
                "token_cost_penalty": 0.0,
            }
        },
    }
    anchors, trace = representation_guided_anchors(
        items,
        val,
        3,
        2,
        12,
        cfg,
        "Rubric",
        backend=FakeBackend(),
        out_dir=tmp_path,
    )
    assert len(anchors) == 3
    assert len(trace) == 3
    assert all(row["representation_mode"] == "local_hidden" for row in trace)


def test_stratified_rep_guided_enforces_score_band_quota(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "anchor_budget": {
            "representation": {
                "mode": "tfidf",
                "max_candidates_per_score": 3,
                "token_cost_penalty": 0.0,
            }
        },
    }
    anchors, trace = stratified_rep_guided_anchors(
        items,
        val,
        9,
        2,
        12,
        cfg,
        "Rubric",
        backend=None,
        out_dir=tmp_path,
    )
    assert len(anchors) == 9
    assert len(trace) == 9
    bands = [row["requested_band"] for row in trace]
    assert bands.count("low") == 3
    assert bands.count("mid") == 3
    assert bands.count("high") == 3
    assert all(a.selection_reason.startswith("stratified_rep_guided:") for a in anchors)


def test_retrieval_grounded_rep_uses_retrieval_candidates_within_each_band(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "anchor_budget": {
            "representation": {"mode": "tfidf", "token_cost_penalty": 0.0},
            "retrieval_grounded_rep": {
                "per_band_top_n": 2,
                "retrieval_weight": 0.6,
                "representation_weight": 0.4,
                "fallback_epsilon": -1.0,
            },
        },
    }
    anchors, trace = retrieval_grounded_stratified_rep_anchors(
        items, val, 6, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path
    )
    assert len(anchors) == 6
    assert len(trace) == 6
    for row in trace:
        assert row["essay_id"] in row["retrieval_candidate_ids_in_band"]
        assert row["retrieval_rank"] <= 2


def test_retrieval_grounded_rep_satisfies_score_band_quota(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "anchor_budget": {
            "representation": {"mode": "tfidf", "token_cost_penalty": 0.0},
            "retrieval_grounded_rep": {"per_band_top_n": 3, "fallback_epsilon": -1.0},
        }
    }
    _, trace = retrieval_grounded_stratified_rep_anchors(
        items, val, 9, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path
    )
    slots = [row["requested_slot"] for row in trace]
    assert slots.count("low_tail") == 1
    assert slots.count("lower_mid") == 1
    assert slots.count("median") == 2
    assert slots.count("upper_mid") == 2
    assert slots.count("high_tail") == 2
    assert slots.count("max_or_top") == 1
    assert all("requested_band" in row for row in trace)


def test_retrieval_grounded_rep_fallback_prefers_retrieval_when_margin_small(tmp_path):
    class FakeBackend:
        def encode_scoring_context(self, **kwargs):
            text = kwargs.get("essay_text", "")
            if "repbest" in text:
                return torch.tensor([1.0, 0.0], dtype=torch.float32)
            return torch.tensor([0.0, 1.0], dtype=torch.float32)

    items = [
        {"essay_id": 1, "essay_text": "retrieval anchor low", "domain1_score": 2},
        {"essay_id": 2, "essay_text": "repbest anchor low", "domain1_score": 2},
        {"essay_id": 3, "essay_text": "other anchor low", "domain1_score": 2},
    ]
    val = [
        {"essay_id": 1001, "essay_text": "retrieval repbest validation", "domain1_score": 2},
    ]
    cfg = {
        "anchor_budget": {
            "representation": {"mode": "local_hidden", "token_cost_penalty": 0.0},
            "retrieval_grounded_rep": {
                "per_band_top_n": 3,
                "retrieval_weight": 0.0,
                "representation_weight": 1.0,
                "fallback_epsilon": 999.0,
            },
        }
    }
    _, trace = retrieval_grounded_stratified_rep_anchors(
        items, val, 1, 2, 12, cfg, "Rubric", backend=FakeBackend(), out_dir=tmp_path
    )
    assert all(row["retrieval_rank"] == 1 for row in trace)
    assert all(row["whether_fallback_to_retrieval"] in {True, False} for row in trace)
    assert any(row["whether_fallback_to_retrieval"] for row in trace)


def test_retrieval_grounded_rep_logging_fields_exist(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {"anchor_budget": {"representation": {"mode": "tfidf"}}}
    _, trace = retrieval_grounded_stratified_rep_anchors(
        items, val, 3, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path
    )
    required = {
        "anchor_id",
        "score",
        "band",
        "retrieval_rank",
        "retrieval_score",
        "representation_score",
        "combined_score",
        "redundancy_score",
        "token_length",
        "selected_reason",
        "whether_selected_by_rep_rerank",
        "whether_fallback_to_retrieval",
    }
    assert required <= set(trace[0])


def test_retrieval_grounded_v21_weights_are_configurable(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "anchor_budget": {
            "representation": {"mode": "tfidf", "token_cost_penalty": 0.0},
            "retrieval_grounded_rep_v21": {
                "per_band_top_n": 3,
                "retrieval_weight": 0.9,
                "representation_weight": 0.1,
            },
        }
    }
    _, trace = retrieval_grounded_stratified_rep_anchors_v21(
        items, val, 3, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path
    )
    assert trace[0]["retrieval_weight"] == pytest.approx(0.9)
    assert trace[0]["representation_weight"] == pytest.approx(0.1)


def test_retrieval_grounded_v21_fallback_margin_blocks_small_rep_gain(tmp_path):
    class FakeBackend:
        def encode_scoring_context(self, **kwargs):
            text = kwargs.get("essay_text", "")
            if "repbest" in text:
                return torch.tensor([1.0, 0.0], dtype=torch.float32)
            return torch.tensor([0.0, 1.0], dtype=torch.float32)

    items = [
        {"essay_id": 1, "essay_text": "retrieval anchor low", "domain1_score": 2},
        {"essay_id": 2, "essay_text": "repbest anchor low", "domain1_score": 2},
    ]
    val = [{"essay_id": 1001, "essay_text": "retrieval repbest validation", "domain1_score": 2}]
    cfg = {
        "anchor_budget": {
            "representation": {"mode": "local_hidden", "token_cost_penalty": 0.0},
            "retrieval_grounded_rep_v21": {
                "per_band_top_n": 2,
                "retrieval_weight": 0.0,
                "representation_weight": 1.0,
                "fallback_margin": 999.0,
                "max_rep_replacements": 3,
            },
        }
    }
    _, trace = retrieval_grounded_stratified_rep_anchors_v21(
        items, val, 1, 2, 12, cfg, "Rubric", backend=FakeBackend(), out_dir=tmp_path
    )
    assert trace[0]["retrieval_rank"] == 1
    assert trace[0]["fallback_to_retrieval"] is True
    assert trace[0]["fallback_reason"] == "small_rerank_margin"


def test_retrieval_grounded_v21_max_rep_replacements_is_enforced(tmp_path):
    class FakeBackend:
        def encode_scoring_context(self, **kwargs):
            text = kwargs.get("essay_text", "")
            if "repbest" in text:
                return torch.tensor([1.0, 0.0], dtype=torch.float32)
            return torch.tensor([0.0, 1.0], dtype=torch.float32)

    items = [
        {"essay_id": 1, "essay_text": "retrieval low a", "domain1_score": 2},
        {"essay_id": 2, "essay_text": "repbest low b", "domain1_score": 2},
        {"essay_id": 3, "essay_text": "retrieval mid a", "domain1_score": 7},
        {"essay_id": 4, "essay_text": "repbest mid b", "domain1_score": 7},
        {"essay_id": 5, "essay_text": "retrieval high a", "domain1_score": 12},
        {"essay_id": 6, "essay_text": "repbest high b", "domain1_score": 12},
    ]
    val = [{"essay_id": 1001, "essay_text": "retrieval repbest validation", "domain1_score": 7}]
    cfg = {
        "anchor_budget": {
            "representation": {"mode": "local_hidden", "token_cost_penalty": 0.0},
            "retrieval_grounded_rep_v21": {
                "per_band_top_n": 2,
                "retrieval_weight": 0.0,
                "representation_weight": 1.0,
                "fallback_margin": -1.0,
                "max_rep_replacements": 1,
            },
        }
    }
    _, trace = retrieval_grounded_stratified_rep_anchors_v21(
        items, val, 3, 2, 12, cfg, "Rubric", backend=FakeBackend(), out_dir=tmp_path
    )
    assert sum(bool(row["selected_by_rep_rerank"]) for row in trace) <= 1
    assert all(row["max_rep_replacements"] == 1 for row in trace)
    assert any(row["fallback_reason"] == "max_rep_replacements_reached" for row in trace)


def test_retrieval_grounded_v21_band_quota_and_no_test_leakage(tmp_path):
    items = _toy_items()
    test_ids = {9001, 9002}
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "anchor_budget": {
            "representation": {"mode": "tfidf", "token_cost_penalty": 0.0},
            "retrieval_grounded_rep_v21": {"per_band_top_n": 3},
        }
    }
    anchors, trace = retrieval_grounded_stratified_rep_anchors_v21(
        items, val, 9, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path
    )
    assert len(anchors) == 9
    slots = [row["requested_slot"] for row in trace]
    assert slots.count("low_tail") == 1
    assert slots.count("lower_mid") == 1
    assert slots.count("median") == 2
    assert slots.count("upper_mid") == 2
    assert slots.count("high_tail") == 2
    assert slots.count("max_or_top") == 1
    assert all("requested_band" in row for row in trace)
    assert not ({a.essay_id for a in anchors} & test_ids)


def test_retrieval_grounded_v21_logging_fields_exist(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {"anchor_budget": {"representation": {"mode": "tfidf"}}}
    _, trace = retrieval_grounded_stratified_rep_anchors_v21(
        items, val, 3, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path
    )
    required = {
        "anchor_id",
        "score",
        "band",
        "retrieval_rank",
        "retrieval_score",
        "representation_score",
        "combined_score",
        "selected_by_rep_rerank",
        "fallback_to_retrieval",
        "fallback_reason",
        "replacement_index",
        "max_rep_replacements",
        "margin_to_retrieval_top",
        "redundancy_score",
        "token_length",
    }
    assert required <= set(trace[0])


def test_retrieval_grounded_no_rep_disables_representation_backend_and_rerank(tmp_path):
    class FailingBackend:
        def encode_scoring_context(self, **kwargs):
            raise AssertionError("no-rep selector must not encode representation")

    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "anchor_budget": {
            "representation": {"mode": "local_hidden"},
            "retrieval_grounded_no_rep": {"per_band_top_n": 3},
        }
    }
    anchors, trace = retrieval_grounded_no_rep_anchors(
        items, val, 9, 2, 12, cfg, "Rubric", backend=FailingBackend(), out_dir=tmp_path
    )
    assert len(anchors) == 9
    assert all(row["method_has_representation"] is False for row in trace)
    assert all(row["representation_score"] == 0.0 for row in trace)
    assert not any(row["selected_by_rep_rerank"] for row in trace)


def test_retrieval_grounded_no_rep_preserves_band_quota_and_logging(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "anchor_budget": {
            "representation": {"mode": "tfidf"},
            "retrieval_grounded_no_rep": {"per_band_top_n": 3},
        }
    }
    anchors, trace = retrieval_grounded_no_rep_anchors(
        items, val, 9, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path
    )
    slots = [row["requested_slot"] for row in trace]
    assert slots.count("low_tail") == 1
    assert slots.count("lower_mid") == 1
    assert slots.count("median") == 2
    assert slots.count("upper_mid") == 2
    assert slots.count("high_tail") == 2
    assert slots.count("max_or_top") == 1
    assert all("requested_band" in row for row in trace)
    assert all(a.selection_reason.startswith("retrieval_grounded_no_rep") for a in anchors)
    required = {
        "anchor_id",
        "score",
        "band",
        "retrieval_rank",
        "retrieval_score",
        "representation_score",
        "combined_score",
        "fallback_to_retrieval",
        "fallback_reason",
        "method_has_representation",
        "redundancy_score",
        "token_length",
    }
    assert required <= set(trace[0])


def test_anchor_stability_estimator_outputs_frequency_and_rank_variance():
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "weak vague limited", "domain1_score": 2},
        {"essay_id": 1003, "essay_text": "middle evidence organized", "domain1_score": 7},
    ]
    rows, trace = estimate_anchor_stability(
        items,
        val,
        k=9,
        score_min=2,
        score_max=12,
        n_bootstrap=4,
        sample_ratio=0.75,
        seed=7,
        per_band_top_n=3,
    )

    assert rows
    assert trace
    assert {"selection_frequency", "mean_rank", "rank_variance", "stability_score"} <= set(rows[0])
    assert all(0.0 <= row["selection_frequency"] <= 1.0 for row in rows)


def test_stability_retrieval_selector_satisfies_band_quota_and_no_test_leakage(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {"anchor_budget": {"stability_retrieval": {"n_bootstrap": 4, "per_band_top_n": 3}}}
    anchors, trace = stability_retrieval_anchors(
        items, val, 9, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path
    )

    assert len(anchors) == 9
    slots = [row["requested_slot"] for row in trace]
    assert slots.count("low_tail") == 1
    assert slots.count("lower_mid") == 1
    assert slots.count("median") == 2
    assert slots.count("upper_mid") == 2
    assert slots.count("high_tail") == 2
    assert slots.count("max_or_top") == 1
    assert all("requested_band" in row for row in trace)
    assert not ({a.essay_id for a in anchors} & {1001, 1002, 1003})
    assert (tmp_path / "anchor_stability_scores.csv").exists()
    assert (tmp_path / "stability_trace.jsonl").exists()


def test_select_stable_anchor_rows_prefers_stability_with_coverage():
    items = _toy_items()
    rows, _ = estimate_anchor_stability(
        items,
        [
            {"essay_id": 1001, "essay_text": "excellent evidence organization language", "domain1_score": 12},
            {"essay_id": 1002, "essay_text": "weak vague limited", "domain1_score": 2},
            {"essay_id": 1003, "essay_text": "middle evidence organized", "domain1_score": 7},
        ],
        k=6,
        score_min=2,
        score_max=12,
        n_bootstrap=3,
        seed=42,
    )
    selected, trace = select_stable_anchor_rows(items, rows, k=6, score_min=2, score_max=12)

    assert len(selected) == 6
    assert len(trace) == 6
    assert {"low", "mid", "high"} <= {row["band"] for row in trace}


def test_select_stable_anchor_rows_can_force_top_tail_coverage():
    items = _toy_items()
    rows = []
    for item in items:
        score = int(item["domain1_score"])
        rows.append(
            {
                "essay_id": int(item["essay_id"]),
                "gold_score": score,
                "band": "low" if score <= 5 else "high" if score >= 9 else "mid",
                "selection_frequency": 1.0 if score == 10 else 0.5,
                "mean_rank": 1.0,
                "rank_variance": 0.0,
                "mean_retrieval_score": 1.0,
                "redundancy_score": 0.0,
                "stability_score": 1.0 if score == 10 else 0.5,
                "selected_count": 1,
                "bootstrap_count": 1,
                "token_length": 10,
            }
        )

    selected, trace = select_stable_anchor_rows(
        items,
        rows,
        k=9,
        score_min=2,
        score_max=12,
        tail_coverage_enabled=True,
        min_top_score_anchors=1,
        top_score_margin=0,
    )

    assert len(selected) == 9
    assert any(int(row["gold_score"]) == 12 for row in selected)
    assert any(row["selected_reason"] == "stability_retrieval: top-tail coverage" for row in trace)


def test_sisa_selector_uses_loo_influence_to_change_ranking():
    items = _toy_items()
    rows = []
    for item in items:
        score = int(item["domain1_score"])
        rows.append(
            {
                "essay_id": int(item["essay_id"]),
                "gold_score": score,
                "band": "low" if score <= 5 else "high" if score >= 9 else "mid",
                "score_slot": "median",
                "selection_frequency": 0.5,
                "mean_rank": 2.0,
                "rank_variance": 0.0,
                "mean_retrieval_score": 1.0,
                "redundancy_score": 0.0,
                "stability_score": 0.5,
                "selected_count": 1,
                "bootstrap_count": 1,
                "token_length": 10,
            }
        )
    # Two median candidates: essay 15 has weaker stability but LOO says it is useful.
    for row in rows:
        if row["essay_id"] == 15:
            row["score_slot"] = "median"
            row["stability_score"] = 0.20
        if row["essay_id"] == 16:
            row["score_slot"] = "median"
            row["stability_score"] = 0.60
    loo_rows = [{"anchor_id": 15, "delta_qwk_without_anchor": -0.30, "delta_mae_without_anchor": 0.20}]

    selected, trace = select_sisa_anchor_rows(
        items,
        rows,
        loo_rows,
        k=1,
        score_min=2,
        score_max=12,
        stability_weight=0.1,
        retrieval_weight=0.0,
        influence_weight=1.0,
        redundancy_weight=0.0,
        token_weight=0.0,
        tail_coverage_enabled=False,
    )

    assert selected[0]["essay_id"] == 15
    assert trace[0]["loo_available"] is True
    assert trace[0]["loo_influence_score"] > 0


def test_sisa_selector_keeps_singleton_top_tail_anchor():
    items = _toy_items()
    rows, _ = estimate_anchor_stability(
        items,
        [{"essay_id": 1001, "essay_text": "excellent evidence organization language", "domain1_score": 12}],
        k=9,
        score_min=2,
        score_max=12,
        n_bootstrap=3,
    )
    selected, trace = select_sisa_anchor_rows(
        items,
        rows,
        [],
        k=9,
        score_min=2,
        score_max=12,
        tail_coverage_enabled=True,
        min_top_score_anchors=1,
        top_score_margin=0,
    )

    assert len(selected) == 9
    assert any(int(row["gold_score"]) == 12 for row in selected)
    assert any(row["selected_reason"] == "sisa: protected top-tail scale anchor" for row in trace)


def test_sisa_anchors_outputs_complete_trace_without_real_llm(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "anchor_budget": {
            "stability_retrieval": {"n_bootstrap": 3, "per_slot_top_n": 3},
            "sisa": {"loo_parent_selection_enabled": True, "loo_max_anchors": 3, "loo_max_items": 3},
        },
        "bapr": {"val_diag_ratio": 0.5},
    }
    anchors, trace = sisa_anchors(items, val, 9, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path)

    assert len(anchors) == 9
    assert trace
    required = {
        "retrieval_score_normalized",
        "stability_score",
        "loo_influence_score",
        "score_slot",
        "redundancy_score",
        "token_cost_penalty",
        "selection_parts",
    }
    assert required <= set(trace[0])
    assert (tmp_path / "sisa_initial_anchor_bank.json").exists()
    assert (tmp_path / "sisa_loo_influence_scores.csv").exists()


def test_sisa_run_one_uses_v_diag_not_full_val_or_test(monkeypatch, tmp_path):
    train = _toy_items()[:12]
    val = [
        {"essay_id": 100 + i, "essay_text": f"val {score}", "domain1_score": score}
        for i, score in enumerate([2, 2, 7, 7, 12, 12])
    ]
    test = [{"essay_id": 900 + i, "essay_text": f"test {score}", "domain1_score": score} for i, score in enumerate([2, 7, 12])]
    expected_diag, expected_sel, _ = runner.split_val_diag_sel(val, 2, 12, 0.5)
    captured = {}

    def fake_sisa(train_rows, val_rows, k, score_min, score_max, config, instruction, backend, out_dir):
        captured["val_ids"] = [int(row["essay_id"]) for row in val_rows]
        anchors = [
            runner.AnchorRecord(
                essay_id=int(row["essay_id"]),
                gold_score=int(row["domain1_score"]),
                prompt_id=1,
                token_length=8,
                source_split="train",
                selection_score=1.0,
                selection_reason="fake_sisa",
                essay_text=row["essay_text"],
            )
            for row in train_rows[: int(k)]
        ]
        return anchors, [{"essay_id": anchor.essay_id, "score_slot": "toy"} for anchor in anchors]

    monkeypatch.setattr(runner, "sisa_anchors", fake_sisa)
    summary = runner.run_one(
        method="sisa_k_anchor",
        k=3,
        config={"data": {"score_min": 2, "score_max": 12, "essay_set": 1}, "bapr": {"val_diag_ratio": 0.5}},
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

    assert captured["val_ids"] == [int(row["essay_id"]) for row in expected_diag]
    assert set(captured["val_ids"]).isdisjoint({int(row["essay_id"]) for row in expected_sel})
    assert set(captured["val_ids"]).isdisjoint({int(row["essay_id"]) for row in test})
    assert (Path(summary["exp_dir"]) / "sisa_val_split.json").exists()


def test_target_score_ladder_adapts_without_hardcoded_p1_scores():
    p1 = target_score_ladder(range(2, 13), 9, score_min=2, score_max=12, strategy="auto")
    p2 = target_score_ladder(range(1, 7), 9)
    p7 = target_score_ladder(range(0, 31), 9)

    assert p1 == [2, 5, 6, 7, 8, 9, 10, 11, 12]
    assert p2 == [1, 2, 3, 4, 5, 6]
    assert p7[0] == 0 and p7[-1] == 30 and len(p7) == 9
    assert 12 not in p7


def test_target_score_ladder_auto_uses_tail_dense_for_near_budget_scales():
    supported = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ladder = target_score_ladder(supported, 9, score_min=2, score_max=12, strategy="auto")

    assert ladder == [2, 5, 6, 7, 8, 9, 10, 11, 12]


def test_coverage_first_sisa_prefers_exact_target_score_before_adjacent():
    items = [
        {"essay_id": 1, "essay_text": "target exact weak", "domain1_score": 5},
        {"essay_id": 2, "essay_text": "adjacent strong excellent evidence", "domain1_score": 6},
    ]
    rows = [
        {
            "essay_id": 1,
            "gold_score": 5,
            "band": "mid",
            "score_slot": "lower_mid",
            "selection_frequency": 0.1,
            "mean_rank": 9.0,
            "rank_variance": 0.0,
            "mean_retrieval_score": 1.0,
            "redundancy_score": 0.0,
            "stability_score": 0.1,
            "selected_count": 0,
            "bootstrap_count": 1,
            "token_length": 10,
        },
        {
            "essay_id": 2,
            "gold_score": 6,
            "band": "mid",
            "score_slot": "median",
            "selection_frequency": 1.0,
            "mean_rank": 1.0,
            "rank_variance": 0.0,
            "mean_retrieval_score": 100.0,
            "redundancy_score": 0.0,
            "stability_score": 1.0,
            "selected_count": 1,
            "bootstrap_count": 1,
            "token_length": 10,
        },
    ]

    selected, trace = select_coverage_first_sisa_anchor_rows(
        items,
        rows,
        [],
        k=1,
        score_min=2,
        score_max=12,
        supported_scores=[5],
        retrieval_weight=1.0,
        stability_weight=1.0,
        influence_weight=0.0,
        diversity_weight=0.0,
        token_weight=0.0,
    )

    assert selected[0]["essay_id"] == 1
    assert trace[0]["target_score"] == 5
    assert trace[0]["exact_score_match"] is True


def test_coverage_first_sisa_adjacent_fallback_is_recorded():
    items = [
        {"essay_id": 1, "essay_text": "near below", "domain1_score": 4},
        {"essay_id": 2, "essay_text": "far above", "domain1_score": 7},
    ]
    rows = []
    for item in items:
        rows.append(
            {
                "essay_id": int(item["essay_id"]),
                "gold_score": int(item["domain1_score"]),
                "band": "mid",
                "score_slot": "median",
                "selection_frequency": 1.0,
                "mean_rank": 1.0,
                "rank_variance": 0.0,
                "mean_retrieval_score": 1.0,
                "redundancy_score": 0.0,
                "stability_score": 1.0,
                "selected_count": 1,
                "bootstrap_count": 1,
                "token_length": 10,
            }
        )

    selected, trace = select_coverage_first_sisa_anchor_rows(
        items,
        rows,
        [],
        k=1,
        score_min=2,
        score_max=12,
        supported_scores=[5],
    )

    assert selected[0]["gold_score"] == 4
    assert trace[0]["exact_score_match"] is False
    assert trace[0]["coverage_gap"] == -1
    assert trace[0]["fallback_reason"] == "adjacent_score_fallback"


def test_coverage_first_sisa_uses_local_influence_without_breaking_ladder():
    items = [
        {"essay_id": 1, "essay_text": "score five stable", "domain1_score": 5},
        {"essay_id": 2, "essay_text": "score five useful", "domain1_score": 5},
        {"essay_id": 3, "essay_text": "score seven unrelated", "domain1_score": 7},
    ]
    rows = []
    for item in items:
        rows.append(
            {
                "essay_id": int(item["essay_id"]),
                "gold_score": int(item["domain1_score"]),
                "band": "mid",
                "score_slot": "median",
                "selection_frequency": 1.0,
                "mean_rank": 1.0,
                "rank_variance": 0.0,
                "mean_retrieval_score": 1.0 if int(item["essay_id"]) != 1 else 2.0,
                "redundancy_score": 0.0,
                "stability_score": 0.5,
                "selected_count": 1,
                "bootstrap_count": 1,
                "token_length": 10,
            }
        )

    selected, trace = select_coverage_first_sisa_anchor_rows(
        items,
        rows,
        [{"anchor_id": 2, "delta_qwk_without_anchor": -0.8, "delta_mae_without_anchor": 0.0}],
        k=1,
        score_min=2,
        score_max=12,
        supported_scores=[5],
        retrieval_weight=0.1,
        stability_weight=0.0,
        influence_weight=1.0,
        diversity_weight=0.0,
        token_weight=0.0,
    )

    assert selected[0]["essay_id"] == 2
    assert selected[0]["gold_score"] == 5
    assert trace[0]["stability_or_influence_changed_local_choice"] is True


def test_coverage_first_sisa_anchors_outputs_complete_trace_without_real_llm(tmp_path):
    items = _toy_items()
    val = [
        {"essay_id": 1001, "essay_text": "excellent evidence organization language sophisticated", "domain1_score": 12},
        {"essay_id": 1002, "essay_text": "middle organized evidence", "domain1_score": 7},
        {"essay_id": 1003, "essay_text": "weak vague limited", "domain1_score": 2},
    ]
    cfg = {
        "anchor_budget": {
            "stability_retrieval": {"n_bootstrap": 3, "per_slot_top_n": 3},
            "coverage_first_sisa": {"loo_enabled": True, "loo_max_anchors": 3, "loo_max_items": 3},
        },
        "bapr": {"val_diag_ratio": 0.5},
    }
    anchors, trace = coverage_first_sisa_anchors(items, val, 9, 2, 12, cfg, "Rubric", backend=None, out_dir=tmp_path)

    assert len(anchors) == 9
    assert trace
    assert all("target_score" in row for row in trace if row.get("selected_reason") != "coverage_first_sisa: no candidate available")
    assert (tmp_path / "coverage_first_sisa_initial_anchor_bank.json").exists()
    assert (tmp_path / "coverage_first_sisa_loo_influence_scores.csv").exists()


def test_coverage_first_sisa_run_one_uses_v_diag_not_full_val_or_test(monkeypatch, tmp_path):
    train = _toy_items()[:12]
    val = [
        {"essay_id": 100 + i, "essay_text": f"val {score}", "domain1_score": score}
        for i, score in enumerate([2, 2, 7, 7, 12, 12])
    ]
    test = [{"essay_id": 900 + i, "essay_text": f"test {score}", "domain1_score": score} for i, score in enumerate([2, 7, 12])]
    expected_diag, expected_sel, _ = runner.split_val_diag_sel(val, 2, 12, 0.5)
    captured = {}

    def fake_coverage(train_rows, val_rows, k, score_min, score_max, config, instruction, backend, out_dir):
        captured["val_ids"] = [int(row["essay_id"]) for row in val_rows]
        anchors = [
            runner.AnchorRecord(
                essay_id=int(row["essay_id"]),
                gold_score=int(row["domain1_score"]),
                prompt_id=1,
                token_length=8,
                source_split="train",
                selection_score=1.0,
                selection_reason="fake_coverage_first_sisa",
                essay_text=row["essay_text"],
            )
            for row in train_rows[: int(k)]
        ]
        return anchors, [{"essay_id": anchor.essay_id, "target_score": anchor.gold_score} for anchor in anchors]

    monkeypatch.setattr(runner, "coverage_first_sisa_anchors", fake_coverage)
    summary = runner.run_one(
        method="coverage_first_sisa_k_anchor",
        k=3,
        config={"data": {"score_min": 2, "score_max": 12, "essay_set": 1}, "bapr": {"val_diag_ratio": 0.5}},
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

    assert captured["val_ids"] == [int(row["essay_id"]) for row in expected_diag]
    assert set(captured["val_ids"]).isdisjoint({int(row["essay_id"]) for row in expected_sel})
    assert set(captured["val_ids"]).isdisjoint({int(row["essay_id"]) for row in test})
    assert (Path(summary["exp_dir"]) / "coverage_first_sisa_val_split.json").exists()


def test_phase1_decision_requires_representation_change():
    rows = [
        {"method": "static_k_anchor", "k": 3, "val_qwk": 0.2, "high_recall": 0.1, "SCI": 0.5, "range_coverage": 0.4},
        {
            "method": "representation_guided_k_anchor",
            "k": 3,
            "val_qwk": 0.25,
            "high_recall": 0.1,
            "SCI": 0.5,
            "range_coverage": 0.4,
            "representation_changed_anchor_choice": True,
        },
    ]
    assert phase1_decision(rows)["decision"] == "PASS"
    rows[1]["representation_changed_anchor_choice"] = False
    assert phase1_decision(rows)["decision"] == "INCONCLUSIVE"


def test_anchor_budget_config_loads_and_dry_run_lists_phase1_plan():
    path = Path("configs/anchor_budget_phase1_p1.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["pace"]["final_pace_calibrated"] is False
    assert "representation_guided_k_anchor" in cfg["anchor_budget"]["methods"]
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_anchor_budget_experiment.py",
            "--config",
            str(path),
            "--fold",
            "0",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert ["no_anchor", None] in payload["plan"]
    assert ["representation_guided_k_anchor", 3] in payload["plan"]
    assert ["full_static_anchor", None] in payload["plan"]


def test_bapr_si_config_loads_and_dry_run_lists_plan():
    path = Path("configs/anchor_budget_bapr_si.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["pace"]["final_pace_calibrated"] is False
    assert cfg["anchor_budget"]["methods"] == ["bapr_si_k_anchor"]
    assert cfg["bapr"]["parent_init_method"] == "stability_retrieval_k_anchor"
    assert cfg["bapr"]["influence"]["loo_attribution_enabled"] is True
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_anchor_budget_experiment.py",
            "--config",
            str(path),
            "--fold",
            "0",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert ["bapr_si_k_anchor", 9] in payload["plan"]


def test_sisa_config_loads_and_dry_run_lists_parent_only_plan():
    path = Path("configs/anchor_budget_sisa_p1.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["pace"]["final_pace_calibrated"] is False
    assert "sisa_k_anchor" in cfg["anchor_budget"]["methods"]
    assert "bapr_si_k_anchor" not in cfg["anchor_budget"]["methods"]
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_anchor_budget_experiment.py",
            "--config",
            str(path),
            "--fold",
            "0",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert ["sisa_k_anchor", 9] in payload["plan"]
    assert ["retrieval_k_anchor", 9] in payload["plan"]
    assert ["full_static_anchor", None] in payload["plan"]


def test_coverage_first_sisa_config_loads_and_dry_run_lists_gate_plan():
    path = Path("configs/anchor_budget_coverage_first_sisa_p1.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["pace"]["final_pace_calibrated"] is False
    assert "coverage_first_sisa_k_anchor" in cfg["anchor_budget"]["methods"]
    assert "bapr_si_k_anchor" not in cfg["anchor_budget"]["methods"]
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_anchor_budget_experiment.py",
            "--config",
            str(path),
            "--fold",
            "0",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert ["coverage_first_sisa_k_anchor", 9] in payload["plan"]
    assert ["static_k_anchor", 9] in payload["plan"]
    assert ["full_static_anchor", None] in payload["plan"]


def test_phase2_dry_run_uses_only_k9_and_expected_prompts():
    proc = subprocess.run(
        [sys.executable, "scripts/run_anchor_budget_phase2.py", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    prompts = {item["prompt"] for item in payload["plan"]}
    assert prompts == {1, 2, 7, 8}
    for item in payload["plan"]:
        cmd = item["command"]
        assert "--ks" in cmd
        assert cmd[cmd.index("--ks") + 1] == "9"
        assert "full_static_anchor" in cmd
        assert "stratified_k_anchor" not in cmd


def test_fixed_split_manifest_overrides_debug_seed(tmp_path):
    data_path = tmp_path / "toy.tsv"
    rows = ["essay_id\tessay_set\tessay\tdomain1_score"]
    for i in range(1, 13):
        rows.append(f"{i}\t1\tEssay {i}\t{2 + (i % 3)}")
    data_path.write_text("\n".join(rows), encoding="latin-1")
    manifest = {
        "prompt_id": 1,
        "essay_set": 1,
        "train_ids": [1, 2, 3, 4],
        "val_ids": [5, 6],
        "test_ids": [7, 8],
        "anchor_pool_ids": [1, 2, 3, 4],
        "score_range": [2, 4],
    }
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "data": {
            "asap_path": str(data_path),
            "essay_set": 1,
            "score_min": 2,
            "score_max": 4,
        },
        "debug": {
            "enabled": True,
            "stratified": False,
            "seed": 999,
            "n_train": 4,
            "n_val": 2,
            "n_test": 2,
        },
    }
    train, val, test = load_asap_split(cfg, 0, split_manifest=manifest_path)
    assert [x["essay_id"] for x in train] == manifest["train_ids"]
    assert [x["essay_id"] for x in val] == manifest["val_ids"]
    assert [x["essay_id"] for x in test] == manifest["test_ids"]
    cfg["debug"]["seed"] = 12345
    train2, val2, test2 = load_asap_split(cfg, 0, split_manifest=manifest_path)
    assert [x["essay_id"] for x in train2] == manifest["train_ids"]
    assert [x["essay_id"] for x in val2] == manifest["val_ids"]
    assert [x["essay_id"] for x in test2] == manifest["test_ids"]
    assert not (set(manifest["anchor_pool_ids"]) & set(manifest["test_ids"]))
