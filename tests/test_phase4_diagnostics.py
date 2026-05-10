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

from pace.llm_backend import ScoringResult, build_scoring_prompt
from pace.pace_fitness import PaceFitnessConfig, PaceFitnessEvaluator
from pace.protocol import (
    AnchorBank,
    DiagnosticType,
    MutationOperator,
    ProtocolCandidate,
    canonical_diagnostic_type,
    mutation_operator_for_diagnostic,
)
from wise_aes import (
    EvolutionOptimizer,
    ExperimentManager,
    PromptIndividual,
    apply_mutation_diversity_quota,
    build_mutation_task_instructions,
    choose_mutation_policy,
    choose_final_primary_label,
    compute_high_score_audit,
    mutation_axis_for_type,
    _split_mutation_selection_val,
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


def test_diagnostic_only_pace_skips_calibrator(monkeypatch):
    class FakeBackend:
        hidden_dim = 8

        def __init__(self):
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "representation_tokens": 0}

        def usage_snapshot(self):
            return dict(self.usage)

        def usage_delta(self, before):
            return {k: self.usage.get(k, 0) - before.get(k, 0) for k in self.usage}

        def score(self, req):
            self.usage["prompt_tokens"] += 10
            self.usage["completion_tokens"] += 2
            y_raw = int(req.score_min + (int(req.essay_id) % (req.score_max - req.score_min + 1)))
            return ScoringResult(
                essay_id=req.essay_id,
                y_raw=y_raw,
                raw_text='{"reasoning":"clear boundary case", "final_score": %d}' % y_raw,
                prompt_text="",
                hidden=torch.ones(8) * float(y_raw),
            )

        def encode_scoring_context(self, **kwargs):
            self.usage["representation_tokens"] += 5
            known = kwargs.get("known_score")
            value = float(known if known is not None else len(str(kwargs.get("essay_text", ""))) % 5)
            return torch.ones(8) * value

    cfg = PaceFitnessConfig(
        diagnostic_only_skip_calibrator=True,
        diagnostic_sample_size=2,
    )
    evaluator = PaceFitnessEvaluator(FakeBackend(), cfg, score_min=2, score_max=12)

    def fail_train(*_args, **_kwargs):
        raise AssertionError("calibrator should be skipped in diagnostic-only mode")

    monkeypatch.setattr(evaluator, "_train_calibrator", fail_train)
    protocol = types.SimpleNamespace(
        instruction_text="rubric",
        static_exemplars=[
            {"essay_id": 1, "essay_text": "low", "domain1_score": 2},
            {"essay_id": 2, "essay_text": "mid", "domain1_score": 8},
            {"essay_id": 3, "essay_text": "high", "domain1_score": 12},
        ],
    )
    fitness_items = [
        {"essay_id": 10 + i, "essay_text": f"essay {i}", "domain1_score": 2 + i}
        for i in range(4)
    ]
    out = evaluator.compute_pace_fitness(protocol, calib_items=[], fitness_items=fitness_items)
    assert out["diagnostic_only_skip_calibrator"] is True
    assert out["n_calib"] == 0
    assert out["n_fitness"] == 2
    assert out["calibrator_train_sec"] == 0.0


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


def test_contrastive_anchor_prompt_and_signature():
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "evolution": {"max_score_contract_enabled": False},
    }
    pair = {
        "boundary": "6_vs_7",
        "lower_anchor": {"essay_id": 11, "score": 6, "essay_text": "lower essay"},
        "upper_anchor": {"essay_id": 12, "score": 7, "essay_text": "upper essay"},
        "rationale_diff": "upper gives clearer evidence",
    }
    ind = PromptIndividual(
        "Rubric body",
        [{"essay_id": 1, "essay_text": "anchor", "domain1_score": 2}],
        config=cfg,
        contrastive_anchors=[pair],
    )
    formatted = ind._format_contrastive_pairs(ind.contrastive_anchors)
    assert "Boundary Pair" in formatted
    assert "6_vs_7" in formatted
    prompt = build_scoring_prompt(
        instruction=ind.scoring_instruction_text(),
        static_exemplars=ind.static_exemplars,
        contrastive_anchors=ind.contrastive_anchors,
        essay_text="target essay",
        score_min=2,
        score_max=12,
    )
    assert "CONTRASTIVE BOUNDARY ANCHORS" in prompt
    assert "upper gives clearer evidence" in prompt
    ind2 = PromptIndividual("Rubric body", ind.static_exemplars, config=cfg, contrastive_anchors=[])
    assert ind.get_signature() != ind2.get_signature()


def test_feedback_uses_actual_eval_items(monkeypatch):
    captured = {}

    def fake_call_llm(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return "feedback"

    monkeypatch.setattr("wise_aes.LOCAL_BACKEND", None)
    monkeypatch.setattr("wise_aes.call_llm", fake_call_llm)
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "evolution": {"n_error_cases": 2},
        "llm": {"temperature_induce": 0.0},
    }
    ind = PromptIndividual("Rubric body", [], config=cfg)
    ind.last_pred_scores = [7]
    ind.last_eval_items = [{"essay_id": 99, "essay_text": "actual evaluated essay", "domain1_score": 12}]
    wrong_full_val = [{"essay_id": 1, "essay_text": "wrong first full-val essay", "domain1_score": 2}]
    assert ind.generate_real_feedback(wrong_full_val) == "feedback"
    assert "ID 99" in captured["prompt"]
    assert "wrong first full-val essay" not in captured["prompt"]


def test_feedback_high_band_text_is_scale_generic(monkeypatch):
    captured = {}

    def fake_call_llm(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return "feedback"

    monkeypatch.setattr("wise_aes.LOCAL_BACKEND", None)
    monkeypatch.setattr("wise_aes.call_llm", fake_call_llm)
    cfg = {
        "data": {"score_min": 0, "score_max": 60},
        "evolution": {"n_error_cases": 1},
        "llm": {"temperature_induce": 0.0},
    }
    ind = PromptIndividual("Rubric body", [], config=cfg)
    ind.last_pred_scores = [50]
    ind.last_eval_items = [{"essay_id": 99, "essay_text": "strong essay", "domain1_score": 60}]
    assert ind.generate_real_feedback([]) == "feedback"
    assert "adaptive high band" in captured["prompt"]
    assert "10-60" not in captured["prompt"]


def test_staged_validation_elite_selection_respects_full_eval_pool():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    cfg = {"data": {"score_min": 2, "score_max": 12}}
    opt.population = [
        PromptIndividual("mini winner", [{"essay_id": 1}], config=cfg),
        PromptIndividual("full candidate", [{"essay_id": 2}], config=cfg),
        PromptIndividual("mid candidate", [{"essay_id": 3}], config=cfg),
    ]
    opt.config = {"evolution": {"elite_diversity_max_score_gap": 10.0}}
    selected = opt._select_elite_indices(
        [0.99, 0.40, 0.95],
        n_elite=1,
        candidate_indices=[1],
    )
    assert selected == [1]


def test_track_candidate_keeps_validation_and_selection_scores_separate():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    cfg = {"data": {"score_min": 2, "score_max": 12}}
    ind = PromptIndividual("rubric", [{"essay_id": 1}], config=cfg)
    opt.best_candidates = {}
    opt.best_candidate_scores = {"best_raw_guarded": float("-inf")}
    opt._track_candidate(
        "best_raw_guarded",
        ind,
        score=0.12,
        gen=1,
        source_idx=0,
        validation_score=0.34,
    )
    snap = opt.best_candidates["best_raw_guarded"]
    assert snap.selection_score == pytest.approx(0.12)
    assert snap.validation_score == pytest.approx(0.34)
    assert snap.fitness == pytest.approx(0.12)


def test_pareto_selection_can_choose_more_balanced_candidate():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    opt.config = {
        "evolution": {
            "pareto_qwk_weight": 1.0,
            "pareto_raw_adjusted_weight": 0.35,
            "pareto_mae_weight": 0.20,
            "pareto_tv_weight": 0.20,
            "pareto_high_recall_weight": 0.15,
            "pareto_max_recall_weight": 0.0,
        }
    }
    snapshots = [
        {
            "raw_prediction_metrics": {
                "mae": 1.60,
                "score_distribution_tv": 0.60,
                "high_score_recall": 0.10,
                "max_score_recall": 0.0,
            }
        },
        {
            "raw_prediction_metrics": {
                "mae": 0.90,
                "score_distribution_tv": 0.10,
                "high_score_recall": 0.70,
                "max_score_recall": 0.0,
            }
        },
    ]
    best_idx, scores = opt._select_pareto_best_index(
        [0, 1],
        raw_scores=[0.56, 0.53],
        raw_adjusted_scores=[0.48, 0.52],
        pop_snapshot=snapshots,
    )
    assert best_idx == 1
    assert scores[1] > scores[0]
    assert "pareto_selection_components" in snapshots[1]


def test_mutation_effect_summary_uses_only_comparable_full_eval_rows():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    rows = [
        {
            "mutation_type": "high_tail_instruction_mutation",
            "delta_comparable": False,
            "delta_raw_qwk": 0.90,
            "delta_high_recall": 0.90,
        },
        {
            "mutation_type": "high_tail_instruction_mutation",
            "delta_comparable": True,
            "delta_raw_qwk": -0.10,
            "delta_high_recall": 0.20,
            "delta_raw_mae": 0.15,
            "delta_score_distribution_tv": 0.10,
            "mutation_acceptance_status": "rejected",
        },
    ]
    summary = opt._mutation_effect_summary(rows)
    assert summary[0]["n_children"] == 2
    assert summary[0]["n_children_full_eval"] == 1
    assert summary[0]["mean_delta_raw_qwk"] == pytest.approx(-0.10)
    assert summary[0]["mean_delta_raw_mae"] == pytest.approx(0.15)
    assert summary[0]["mean_delta_score_distribution_tv"] == pytest.approx(0.10)
    assert summary[0]["win_rate_vs_parent_raw_qwk"] == pytest.approx(0.0)
    assert summary[0]["win_rate_vs_parent_high_recall"] == pytest.approx(1.0)
    assert summary[0]["accepted_children"] == 0
    assert summary[0]["rejected_children"] == 1
    assert summary[0]["acceptance_rate"] == pytest.approx(0.0)


def test_pace_selection_default_is_diagnostic_only():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    opt.config = {"pace": {}, "data": {"score_min": 2, "score_max": 12}}
    opt.pace_evaluator = types.SimpleNamespace(config=types.SimpleNamespace(gamma=0.1))
    pace_result = {
        "anchor_geometry_score": 1.0,
        "cost_penalty": 0.2,
        "overfit_penalty": 0.2,
        "distribution_penalty": 0.2,
    }
    out = opt._pace_selection_combined(pace_result, 0.33)
    assert out == pytest.approx(0.33)
    assert pace_result["selection_objective"].startswith("raw_val_qwk only")


def test_raw_guard_ignores_pace_distribution_when_diagnostic_only():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    opt.config = {
        "pace": {
            "selection_influence": "diagnostic_only",
            "selection_max_distribution_penalty": 0.15,
        },
        "evolution": {
            "raw_high_recall_floor": 0.30,
            "raw_high_bias_floor": -1.0,
        },
        "data": {"score_min": 2, "score_max": 12},
    }
    pop_snapshot = [
        {
            "raw_prediction_metrics": {
                "high_recall": 1.0,
                "high_pred_bias": 0.0,
            }
        }
    ]
    selection, triggered, feasible = opt._apply_raw_guard(
        raw_scores=[0.50],
        protocol_quality_scores=[0.50],
        pop_snapshot=pop_snapshot,
        raw_adjusted_scores=[0.50],
        pace_results={0: {"distribution_penalty": 1.0, "selection_influence": "diagnostic_only"}},
    )
    assert selection == [pytest.approx(0.50)]
    assert triggered == []
    assert feasible == [0]
    assert "pace_distribution_penalty_high" not in pop_snapshot[0]["constraint_reasons"]


def test_mutation_acceptance_guard_rejects_unbalanced_child():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    opt.config = {
        "evolution": {
            "mutation_acceptance_guard_enabled": True,
            "mutation_acceptance_min_raw_delta": -0.005,
            "mutation_acceptance_min_raw_adjusted_delta": -0.010,
            "mutation_acceptance_max_mae_increase": 0.05,
            "mutation_acceptance_max_tv_increase": 0.05,
            "mutation_acceptance_min_high_recall_delta": -0.05,
            "mutation_acceptance_reject_penalty": 0.03,
            "mutation_acceptance_require_full_comparable": True,
        }
    }
    pop_snapshot = [
        {
            "validation_stage": "full",
            "raw_fitness": 0.49,
            "raw_adjusted_fitness": 0.45,
            "raw_mae": 1.20,
            "raw_distribution_tv": 0.30,
            "high_score_recall": 0.40,
            "max_score_recall": 0.0,
            "parent_trace": {
                "parent_validation_stage": "full",
                "parent_raw_qwk": 0.52,
                "parent_raw_adjusted_qwk": 0.50,
                "parent_raw_mae": 1.05,
                "parent_score_distribution_tv": 0.20,
                "parent_high_score_recall": 0.50,
                "parent_max_score_recall": 0.0,
                "parent_protocol_quality": 0.50,
            },
        }
    ]
    guarded, rejected, accepted = opt._apply_mutation_acceptance_guard([0.49], pop_snapshot)
    assert rejected == [0]
    assert accepted == []
    assert guarded[0] == pytest.approx(0.47)
    assert pop_snapshot[0]["mutation_acceptance_status"] == "rejected"
    assert "raw_qwk_drop" in pop_snapshot[0]["mutation_acceptance_reasons"]
    assert "mae_increase" in pop_snapshot[0]["mutation_acceptance_reasons"]


def test_mutation_acceptance_guard_accepts_balanced_child():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    opt.config = {
        "evolution": {
            "mutation_acceptance_guard_enabled": True,
            "mutation_acceptance_min_raw_delta": -0.005,
            "mutation_acceptance_min_raw_adjusted_delta": -0.010,
            "mutation_acceptance_max_mae_increase": 0.05,
            "mutation_acceptance_max_tv_increase": 0.05,
            "mutation_acceptance_min_high_recall_delta": -0.05,
            "mutation_acceptance_require_full_comparable": True,
        }
    }
    pop_snapshot = [
        {
            "validation_stage": "full",
            "raw_fitness": 0.53,
            "raw_adjusted_fitness": 0.51,
            "raw_mae": 1.04,
            "raw_distribution_tv": 0.19,
            "high_score_recall": 0.55,
            "max_score_recall": 0.0,
            "parent_trace": {
                "parent_validation_stage": "full",
                "parent_raw_qwk": 0.52,
                "parent_raw_adjusted_qwk": 0.50,
                "parent_raw_mae": 1.05,
                "parent_score_distribution_tv": 0.20,
                "parent_high_score_recall": 0.50,
                "parent_max_score_recall": 0.0,
                "parent_protocol_quality": 0.50,
            },
        }
    ]
    guarded, rejected, accepted = opt._apply_mutation_acceptance_guard([0.53], pop_snapshot)
    assert rejected == []
    assert accepted == [0]
    assert guarded[0] == pytest.approx(0.53)
    assert pop_snapshot[0]["mutation_acceptance_status"] == "accepted"


def test_mutation_acceptance_guard_accepts_pareto_tradeoff_child():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    opt.config = {
        "evolution": {
            "mutation_acceptance_guard_enabled": True,
            "mutation_acceptance_mode": "pareto",
            "mutation_acceptance_allow_tradeoff": True,
            "mutation_acceptance_min_tradeoff_improvements": 2,
            "mutation_acceptance_min_raw_delta": -0.005,
            "mutation_acceptance_min_raw_adjusted_delta": -0.010,
            "mutation_acceptance_max_mae_increase": 0.05,
            "mutation_acceptance_max_tv_increase": 0.05,
            "mutation_acceptance_min_high_recall_delta": -0.05,
            "mutation_acceptance_qwk_hard_drop": -0.08,
            "mutation_acceptance_require_full_comparable": True,
        }
    }
    pop_snapshot = [
        {
            "validation_stage": "full",
            "raw_fitness": 0.49,
            "raw_adjusted_fitness": 0.51,
            "raw_mae": 0.95,
            "raw_distribution_tv": 0.12,
            "high_score_recall": 0.60,
            "max_score_recall": 0.0,
            "parent_trace": {
                "parent_validation_stage": "full",
                "parent_raw_qwk": 0.52,
                "parent_raw_adjusted_qwk": 0.50,
                "parent_raw_mae": 1.05,
                "parent_score_distribution_tv": 0.20,
                "parent_high_score_recall": 0.50,
                "parent_max_score_recall": 0.0,
                "parent_protocol_quality": 0.50,
            },
        }
    ]
    guarded, rejected, accepted = opt._apply_mutation_acceptance_guard([0.49], pop_snapshot)
    assert rejected == []
    assert accepted == [0]
    assert guarded[0] == pytest.approx(0.49)
    assert pop_snapshot[0]["mutation_acceptance_status"] == "accepted_tradeoff"
    assert "raw_qwk_drop" in pop_snapshot[0]["mutation_acceptance_reasons"]
    assert pop_snapshot[0]["mutation_acceptance_tradeoff_improvements"] >= 2


def test_acceptance_safe_rank_pool_excludes_rejected_child():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    pop_snapshot = [
        {"mutation_acceptance_status": "no_parent"},
        {"mutation_acceptance_status": "rejected"},
        {"mutation_acceptance_status": "accepted"},
    ]
    assert opt._acceptance_safe_candidate_indices(pop_snapshot, [0, 1, 2]) == [0, 2]


def test_acceptance_safe_rank_pool_falls_back_if_all_rejected():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    pop_snapshot = [
        {"mutation_acceptance_status": "rejected"},
        {"mutation_acceptance_status": "rejected"},
    ]
    assert opt._acceptance_safe_candidate_indices(pop_snapshot, [0, 1]) == [0, 1]


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
    cfg = {"data": {"score_min": 2, "score_max": 12}, "evolution": {"raw_high_recall_floor": 0.30}}
    policy = choose_mutation_policy(
        object(),
        {"dominant_error_type": dominant, "raw_prediction_metrics": {"high_score_recall": 1.0, "max_score_recall": 1.0}},
        cfg,
    )
    assert policy["mutation_type"] == expected
    assert policy["mutation_axis"] == mutation_axis_for_type(expected)


def test_hidden_mutation_priority_overrides_boundary_to_high_tail():
    dummy = type("Dummy", (), {"static_exemplars": [{"essay_id": 1}, {"essay_id": 2}]})()
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "pace": {"trigger_high_recall_floor": 0.5},
        "evolution": {
            "diagnostic_source": "hidden_evidence",
            "raw_high_recall_floor": 0.3,
            "hidden_mutation_priority_enabled": True,
            "hidden_mutation_priority_threshold": 0.1,
        },
    }
    policy = choose_mutation_policy(
        dummy,
        {
            "dominant_error_type": "boundary_ambiguity",
            "pace_diagnostic_summary": {
                "under_score_high_hidden": 8,
                "boundary_ambiguity": 2,
            },
            "raw_prediction_metrics": {
                "high_score_recall": 0.31,
                "max_score_recall": 1.0,
                "high_pred_bias": -1.2,
                "score_distribution_tv": 0.4,
                "pred_collapse_ratio": 0.2,
            },
        },
        cfg,
    )
    assert policy["raw_mutation_type"] == "boundary_clarification_mutation"
    assert policy["mutation_type"] == "high_tail_instruction_mutation"
    assert policy["hidden_priority_changed_decision"] is True
    assert policy["selected_by_raw_or_hidden"] == "hidden_priority"
    assert policy["hidden_repair_score"] > 0.1


def test_protocol_candidate_and_diagnostic_taxonomy():
    diag = canonical_diagnostic_type(
        "under_score_high_hidden",
        {"high_score_recall": 0.2},
    )
    assert diag == DiagnosticType.HIGH_TAIL_UNDERSCORE
    assert mutation_operator_for_diagnostic(diag) == MutationOperator.HIGH_TAIL_INSTRUCTION
    candidate = ProtocolCandidate(
        id="p1",
        parent_id=None,
        instruction="rubric",
        anchor_bank=AnchorBank(),
        diagnostic_type=diag.value,
        mutation_operator=MutationOperator.HIGH_TAIL_INSTRUCTION.value,
    )
    payload = candidate.to_dict()
    assert payload["id"] == "p1"
    assert payload["diagnostic_type"] == "HIGH_TAIL_UNDERSCORE"
    assert "anchor_bank" in payload


def test_reflection_baseline_forces_general_reflection_policy():
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "evolution": {
            "diagnostic_source": "llm_reflection",
            "raw_high_recall_floor": 0.30,
        },
    }
    policy = choose_mutation_policy(
        object(),
        {
            "dominant_error_type": "anchor_confusion",
            "raw_prediction_metrics": {
                "high_score_recall": 0.0,
                "max_score_recall": 0.0,
                "n_true_max_score": 2,
            },
        },
        cfg,
    )
    assert policy["mutation_type"] == "general_reflection_mutation"
    assert policy["diagnostic_source"] == "llm_reflection"


def test_final_primary_policy_defaults_to_raw_guarded():
    candidates = {
        "best_raw_val": object(),
        "best_raw_guarded": object(),
        "best_pareto": object(),
        "best_protocol_quality": object(),
    }
    assert choose_final_primary_label(candidates, {"evolution": {}}) == "best_raw_guarded"
    assert (
        choose_final_primary_label(
            candidates,
            {"evolution": {"final_primary_candidate": "best_pareto"}},
        )
        == "best_pareto"
    )


def test_max_score_contrastive_policy_and_prompt_are_scale_generic():
    cfg = {"data": {"score_min": 0, "score_max": 60}, "evolution": {"raw_high_recall_floor": 0.30}}
    policy = choose_mutation_policy(
        object(),
        {
            "dominant_error_type": "general_error",
            "raw_prediction_metrics": {
                "high_score_recall": 0.8,
                "max_score_recall": 0.0,
                "n_true_max_score": 2,
            },
        },
        cfg,
    )
    assert policy["mutation_type"] == "max_score_contrastive_mutation"
    task = build_mutation_task_instructions(policy["mutation_type"], cfg)
    assert "60" in task
    assert "12" not in task
    assert "10" not in task


def test_mutation_policy_prefers_dominant_boundary_over_generic_max_gap():
    cfg = {"data": {"score_min": 2, "score_max": 12}, "evolution": {"raw_high_recall_floor": 0.30}}
    policy = choose_mutation_policy(
        object(),
        {
            "dominant_error_type": "boundary_ambiguity",
            "raw_prediction_metrics": {
                "high_score_recall": 0.3,
                "max_score_recall": 0.0,
                "n_true_max_score": 2,
            },
        },
        cfg,
    )
    assert policy["mutation_type"] == "boundary_clarification_mutation"


def test_mutation_diversity_quota_avoids_single_type():
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "evolution": {
            "mutation_diversity_enabled": True,
            "mutation_type_quota": {
                "high_tail_instruction_mutation": 2,
                "anchor_slot_mutation": 1,
            },
        },
    }
    planned = apply_mutation_diversity_quota(
        [{"mutation_type": "high_tail_instruction_mutation"}],
        cfg,
        n_children=3,
    )
    assert len(planned) == 3
    assert len({p["actual_mutation_type"] for p in planned}) > 1
    assert all(p["mutation_axis"] in {"I", "E", "IE"} for p in planned)


def test_mutation_diversity_keeps_diagnostic_primary_first():
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "evolution": {
            "mutation_diversity_enabled": True,
            "mutation_type_quota": {
                "high_tail_instruction_mutation": 1,
                "max_score_contrastive_mutation": 1,
            },
        },
    }
    planned = apply_mutation_diversity_quota(
        [{"mutation_type": "boundary_clarification_mutation"}],
        cfg,
        n_children=2,
    )
    assert planned[0]["actual_mutation_type"] == "boundary_clarification_mutation"


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


def test_candidate_lineage_jsonl_serializes(tmp_path):
    mgr = ExperimentManager.__new__(ExperimentManager)
    mgr.exp_dir = tmp_path
    mgr.config = {
        "evolution": {"diagnostic_source": "hidden_evidence"},
        "data": {"prompt_id": 1},
    }
    population = [
        {
            "signature": "child",
            "static_exemplar_ids": [1, 2],
            "static_exemplar_scores": [8, 12],
            "contrastive_anchor_pair_ids": ["8_vs_9:1-2"],
            "contrastive_anchor_boundaries": ["8_vs_9"],
            "raw_fitness": 0.4,
            "raw_adjusted_fitness": 0.35,
            "selection_fitness": 0.35,
            "protocol_quality": 0.4,
            "validation_stage": "full",
            "validation_n": 16,
            "raw_prediction_metrics": {
                "high_score_recall": 0.5,
                "max_score_recall": 0.25,
                "score_distribution_tv": 0.4,
                "mae": 1.0,
            },
            "protocol_candidate": {
                "id": "child",
                "parent_id": "parent",
                "diagnostic_source": "hidden_evidence",
                "diagnostic_type": "HIGH_TAIL_UNDERSCORE",
                "mutation_operator": "max_score_contrastive_mutation",
            },
        }
    ]
    mgr.save_candidate_lineage(2, population, metrics={"tokens_total_all": 100, "duration_sec": 3.0})
    out = tmp_path / "candidate_lineage.jsonl"
    assert out.exists()
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["generation"] == 2
    assert row["candidate_id"] == "child"
    assert row["parent_id"] == "parent"
    assert row["val_sel_metrics"]["high_score_recall"] == 0.5
    assert row["contrastive_anchor_pair_ids"] == ["8_vs_9:1-2"]


def test_dual_validation_split_is_disjoint_and_stratified():
    val_set = [
        {"essay_id": i, "essay_text": f"essay {i}", "domain1_score": 2 + (i % 11)}
        for i in range(44)
    ]
    cfg = {
        "data": {"score_min": 2, "score_max": 12},
        "evolution": {"dual_validation_enabled": True, "mutation_val_ratio": 0.5},
    }
    mutation_val, selection_val, meta = _split_mutation_selection_val(val_set, cfg, seed=123)
    mutation_ids = {x["essay_id"] for x in mutation_val}
    selection_ids = {x["essay_id"] for x in selection_val}
    assert meta["dual_validation_enabled"] is True
    assert mutation_ids.isdisjoint(selection_ids)
    assert len(mutation_val) + len(selection_val) == len(val_set)
    assert abs(len(mutation_val) - len(selection_val)) <= 1
    assert meta["mutation_selection_overlap"] == 0


def test_contrastive_anchor_pairs_are_score_range_generic():
    opt = EvolutionOptimizer.__new__(EvolutionOptimizer)
    opt.config = {
        "data": {"score_min": 0, "score_max": 60},
        "evolution": {
            "contrastive_anchors_enabled": True,
            "n_contrastive_anchor_pairs": 2,
        },
    }
    opt.train_data = [
        {"essay_id": 1, "essay_text": "score twenty nine", "domain1_score": 29},
        {"essay_id": 2, "essay_text": "score thirty", "domain1_score": 30},
        {"essay_id": 3, "essay_text": "score forty nine", "domain1_score": 49},
        {"essay_id": 4, "essay_text": "score fifty", "domain1_score": 50},
        {"essay_id": 5, "essay_text": "score ten", "domain1_score": 10},
    ]
    opt.anchor_slot_specs = [
        {"name": "boundary_29_30", "min": 29, "max": 30},
        {"name": "boundary_49_50", "min": 49, "max": 50},
    ]
    pairs = opt._build_contrastive_anchor_pairs()
    assert len(pairs) == 2
    assert pairs[0]["boundary"] in {"29_vs_30", "49_vs_50"}
    all_scores = {
        pairs[0]["lower_anchor"]["score"],
        pairs[0]["upper_anchor"]["score"],
        pairs[1]["lower_anchor"]["score"],
        pairs[1]["upper_anchor"]["score"],
    }
    assert 12 not in all_scores


@pytest.mark.parametrize(
    "path",
    [
        "configs/phase4_raw_only_full_fold.yaml",
        "configs/phase4_raw_only_smoke.yaml",
        "configs/phase4_raw_error_pe_smoke.yaml",
        "configs/phase4_reflection_pe_smoke.yaml",
        "configs/phase4_smoke.yaml",
        "configs/phase4_neural_smoke.yaml",
        "configs/phase4_neural_no_anchor_smoke.yaml",
        "configs/phase4_neural_text_anchor_smoke.yaml",
        "configs/phase4_neural_score_anchor_smoke.yaml",
        "configs/phase4_neural_dual_anchor_smoke.yaml",
        "configs/phase4_z_anchor_only_smoke.yaml",
        "configs/phase4_z_no_anchor_smoke.yaml",
        "configs/phase4_cost_aware_mid.yaml",
        "configs/phase4_raw_only_cost_aware_mid.yaml",
        "configs/phase4_fixed_mid.yaml",
        "configs/phase4_raw_error_pe_mid.yaml",
        "configs/phase4_reflection_pe_mid.yaml",
        "configs/phase4_hidden_pe_mid.yaml",
        "configs/phase4_hidden_contrastive_mid.yaml",
        "configs/phase4_gate3_raw_no_anchor_mid.yaml",
        "configs/phase4_gate3_raw_absolute_mid.yaml",
        "configs/phase4_gate3_raw_abs_contrastive_mid.yaml",
        "configs/phase4_gate3_hidden_no_anchor_mid.yaml",
        "configs/phase4_gate3_hidden_absolute_mid.yaml",
        "configs/phase4_gate3_hidden_abs_contrastive_mid.yaml",
        "configs/phase4_gate3_rescue_raw_absolute_mid.yaml",
        "configs/phase4_gate3_rescue_hidden_absolute_mid.yaml",
    ],
)
def test_phase4_configs_load(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert "data" in cfg
    assert "evolution" in cfg
    assert "pace" in cfg
    assert cfg["pace"].get("final_pace_calibrated") is False
