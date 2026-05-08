from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from pace.llm_backend import ScoringResult, resolve_llm_token_limits
from pace.neural_evidence import NeuralEvidenceConfig, NeuralEvidenceEncoder
from pace.pace_fitness import PaceFitnessConfig, PaceFitnessEvaluator


@pytest.mark.parametrize("anchor_view", ["none", "text", "score", "dual"])
def test_neural_evidence_encoder_forward_shape(anchor_view):
    cfg = NeuralEvidenceConfig(
        hidden_input_dim=8,
        proj_dim=8,
        score_emb_dim=4,
        output_dim=16,
        anchor_view=anchor_view,
        num_attention_heads=2,
    )
    enc = NeuralEvidenceEncoder(cfg)
    target = torch.randn(2, 8)
    anchor_text = torch.randn(3, 8)
    anchor_score = torch.randn(3, 8)
    anchor_scores = torch.tensor([2, 8, 12])
    reasoning = torch.randn(2, 8)
    z, aux = enc(
        target_hidden=target,
        anchor_hidden_text=anchor_text,
        anchor_hidden_score=anchor_score,
        anchor_scores=anchor_scores,
        score_min=2,
        score_max=12,
        reasoning_hidden=reasoning,
        y_raw=torch.tensor([8, 10]),
    )
    assert z.shape == (2, 16)
    assert aux["enabled_views"]["anchor_view"] == anchor_view
    if anchor_view == "none":
        assert aux["attention_weights"].shape[-1] == 0


def test_neural_evidence_encoder_without_reasoning_or_score():
    cfg = NeuralEvidenceConfig(
        hidden_input_dim=8,
        proj_dim=8,
        score_emb_dim=4,
        output_dim=12,
        anchor_view="none",
        use_reasoning_hidden=False,
        use_raw_score_embedding=False,
    )
    enc = NeuralEvidenceEncoder(cfg)
    z, _ = enc(
        target_hidden=torch.randn(8),
        score_min=0,
        score_max=6,
    )
    assert z.shape == (1, 12)


def test_neural_evidence_is_deterministic_and_block_preserving():
    cfg = NeuralEvidenceConfig(
        hidden_input_dim=8,
        proj_dim=8,
        score_emb_dim=4,
        output_dim=16,
        anchor_view="dual",
        attention_mode="cosine",
        preserve_block_structure=True,
        projection_seed=123,
        num_attention_heads=2,
    )
    enc_a = NeuralEvidenceEncoder(cfg)
    enc_b = NeuralEvidenceEncoder(cfg)
    target = torch.randn(2, 8)
    anchor_text = torch.randn(3, 8)
    anchor_score = torch.randn(3, 8)
    kwargs = dict(
        target_hidden=target,
        anchor_hidden_text=anchor_text,
        anchor_hidden_score=anchor_score,
        anchor_scores=torch.tensor([2, 8, 12]),
        score_min=2,
        score_max=12,
        y_raw=torch.tensor([8, 10]),
    )
    z_a, aux = enc_a(**kwargs)
    z_b, _ = enc_b(**kwargs)
    assert torch.allclose(z_a, z_b)
    assert z_a.shape[-1] == sum(enc_a.block_dims())
    assert aux["enabled_views"]["attention_mode"] == "cosine"
    assert aux["enabled_views"]["preserve_block_structure"] is True


class _FakeBackend:
    hidden_dim = 8

    def encode_text_mean(self, _text):
        return torch.ones(8)


def test_neural_hidden_does_not_call_manual_feature_builders(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("manual feature builder should not be used")

    monkeypatch.setattr("pace.pace_fitness.build_reasoning_features", fail)
    monkeypatch.setattr("pace.pace_fitness.build_objective_features", fail)
    monkeypatch.setattr("pace.pace_fitness.build_uncertainty_features", fail)

    cfg = PaceFitnessConfig(
        evidence_mode="neural_hidden",
        neural_evidence={
            "hidden_input_dim": 8,
            "proj_dim": 8,
            "score_emb_dim": 4,
            "output_dim": 16,
            "anchor_view": "dual",
            "num_attention_heads": 2,
        },
    )
    evaluator = PaceFitnessEvaluator(_FakeBackend(), cfg, score_min=2, score_max=12)
    result = ScoringResult(
        essay_id=1,
        y_raw=8,
        raw_text='{"reasoning":"reasonable", "final_score":8}',
        prompt_text="",
        hidden=torch.randn(8),
    )
    anchors = torch.randn(3, 8)
    payload = {
        "anchor_hiddens_text": torch.randn(3, 8),
        "anchor_hiddens_score": anchors,
        "anchor_scores": [2, 8, 12],
    }
    z = evaluator._build_evidence_bundle(
        result,
        "Essay text is not used by neural_hidden evidence.",
        anchors,
        anchor_scores=[2, 8, 12],
        anchor_payload=payload,
    )
    assert z.shape == (16,)
    assert evaluator.evidence_metadata(3)["evidence_mode"] == "neural_hidden"


def test_dual_view_config_loads():
    cfg = yaml.safe_load(Path("configs/phase4_neural_dual_anchor_smoke.yaml").read_text(encoding="utf-8"))
    pace_cfg = PaceFitnessConfig.from_config(cfg)
    assert pace_cfg.evidence_mode == "neural_hidden"
    assert pace_cfg.neural_evidence["anchor_view"] == "dual"


def test_backend_token_limit_resolution():
    limits = resolve_llm_token_limits(
        {
            "llm": {
                "max_new_tokens_scoring": 192,
                "max_new_tokens_reflection": 512,
                "max_new_tokens_induction": 384,
            }
        },
        default=768,
    )
    assert limits["scoring"] == 192
    assert limits["reflection"] == 512
    assert limits["induction"] == 384
    assert limits["default"] == 768
