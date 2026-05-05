from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

    stub.SentenceTransformer = _DummySentenceTransformer
    sys.modules["sentence_transformers"] = stub

from pace.llm_backend import ScoringResult
from pace.pace_fitness import PaceFitnessConfig, PaceFitnessEvaluator
from wise_aes import (
    ExperimentManager,
    PromptIndividual,
    choose_mutation_policy,
    compute_high_score_audit,
)


def _pace_eval(blocks=None):
    cfg = PaceFitnessConfig()
    if blocks is not None:
        cfg.evidence_blocks_enabled.update(blocks)
    return PaceFitnessEvaluator(
        local_backend=object(),
        config=cfg,
        score_min=2,
        score_max=12,
    )


def test_evidence_block_mask_preserves_dimension():
    base = _pace_eval()
    masked = _pace_eval({"anchor": False, "reasoning": False})
    result = ScoringResult(
        essay_id=1,
        y_raw=8,
        raw_text='{"reasoning": "clear but somewhat limited", "final_score": 8}',
        prompt_text="",
        hidden=torch.ones(8),
    )
    anchors = torch.stack([torch.zeros(8), torch.ones(8), torch.full((8,), 2.0)], dim=0)
    z_base = base._build_evidence_bundle(result, "This is a short essay.", anchors, [2, 8, 12])
    z_masked = masked._build_evidence_bundle(result, "This is a short essay.", anchors, [2, 8, 12])
    assert z_base.numel() == z_masked.numel()
    slices = masked.evidence_block_slices(3)
    assert torch.allclose(z_masked[slices["anchor"]], torch.zeros(len(slices["anchor"])))
    assert torch.allclose(z_masked[slices["reasoning"]], torch.zeros(len(slices["reasoning"])))


def test_max_score_contract_toggle_changes_scoring_instruction():
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "evolution": {"max_score_contract_enabled": False},
    }
    ind = PromptIndividual("Rubric body", [], config=cfg)
    assert "Maximum-Score Attainable Contract" not in ind.scoring_instruction_text()
    cfg_on = {
        "data": {"score_min": 2, "score_max": 12},
        "evolution": {"max_score_contract_enabled": True},
    }
    ind_on = PromptIndividual("Rubric body", [], config=cfg_on)
    assert "Maximum-Score Attainable Contract" in ind_on.scoring_instruction_text()
    assert "maximum score is attainable" in ind_on.scoring_instruction_text().lower()


def test_high_score_audit_toy_values():
    audit = compute_high_score_audit(
        [2, 8, 10, 12],
        [2, 8, 9, 12],
        score_min=2,
        score_max=12,
        high_score_threshold=10,
    )
    assert audit["n_true_high"] == 2
    assert audit["n_pred_high"] == 1
    assert audit["high_score_recall"] == pytest.approx(0.5)
    assert audit["high_score_precision"] == pytest.approx(1.0)
    assert audit["n_true_max_score"] == 1
    assert audit["n_pred_max_score"] == 1
    assert audit["max_score_recall"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "dominant,expected",
    [
        ("under_score_high_hidden", "high_tail_instruction_mutation"),
        ("over_score_low_hidden", "negative_constraint_mutation"),
        ("anchor_confusion", "anchor_slot_mutation"),
        ("reasoning_score_contradiction", "score_mapping_mutation"),
        ("raw_collapse", "score_distribution_mutation"),
        ("boundary_ambiguity", "boundary_clarification_mutation"),
        ("general_error", "general_reflection_mutation"),
    ],
)
def test_mutation_policy_mapping(dominant, expected):
    cfg = {"evolution": {"raw_high_recall_floor": 0.30}}
    policy = choose_mutation_policy(
        object(),
        {"dominant_error_type": dominant, "raw_prediction_metrics": {"high_score_recall": 1.0, "max_score_recall": 1.0}},
        cfg,
    )
    assert policy["mutation_type"] == expected


def test_parent_child_audit_row_serializes(tmp_path):
    mgr = ExperimentManager.__new__(ExperimentManager)
    mgr.exp_dir = tmp_path
    row = {
        "generation": 2,
        "parent_signature": "p",
        "child_signature": "c",
        "mutation_type": "anchor_slot_mutation",
        "changed_anchor_slots": [1],
        "changed_anchor_ids_before": [10],
        "changed_anchor_ids_after": [20],
        "delta_raw_qwk": 0.01,
    }
    mgr.save_parent_child_audit([row])
    out = tmp_path / "parent_child_audit.csv"
    assert out.exists()
    with out.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["mutation_type"] == "anchor_slot_mutation"
    assert json.loads(rows[0]["changed_anchor_slots"]) == [1]


@pytest.mark.parametrize(
    "path",
    [
        "configs/phase4_raw_only_full_fold.yaml",
        "configs/phase4_raw_only_smoke.yaml",
        "configs/phase4_smoke.yaml",
    ],
)
def test_phase4_configs_load(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert "data" in cfg
    assert "evolution" in cfg
    assert "pace" in cfg
    assert cfg["pace"].get("final_pace_calibrated") is False
