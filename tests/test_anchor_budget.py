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
    phase1_decision,
    representation_guided_anchors,
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
