from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_anchor_budget_experiment import (  # noqa: E402
    deterministic_score_covered,
    load_asap_split,
    phase1_decision,
    representation_guided_anchors,
    retrieval_grounded_stratified_rep_anchors,
    score_boundary_metrics,
    stratified_rep_guided_anchors,
    stratified_anchors,
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
    bands = [row["requested_band"] for row in trace]
    assert bands.count("low") == 3
    assert bands.count("mid") == 3
    assert bands.count("high") == 3


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
