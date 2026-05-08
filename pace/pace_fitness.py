"""
WISE-PACE protocol fitness evaluator.

The evaluator produces hidden-evidence diagnostics for candidate protocols
P=<I,E>. Any calibrated QWK is a diagnostic probe only; protocol selection in
wise_aes.py is raw-first and does not reward pace_qwk.

The default evidence vector is an enhanced compact vector with an explicit
schema:
  raw score features
  + anchor-relative hidden geometry features
  + reasoning-text features
  + objective essay features
  + uncertainty features

Raw 4k hidden states are intentionally not fed to the calibrator, because the
calibration split is usually tiny during evolution.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score

from pace.calibration import CalibratorConfig, CoralOrdinalCalibrator, calibrator_loss
from pace.evidence import (
    build_objective_features,
    build_reasoning_features,
    build_uncertainty_features,
)
from pace.llm_backend import LocalLlamaBackend, ScoringRequest, ScoringResult
from pace.neural_evidence import NeuralEvidenceConfig, NeuralEvidenceEncoder

if TYPE_CHECKING:
    pass  # PromptIndividual 用字符串注解，避免循环导入


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class PaceFitnessConfig:
    top_k_pace: int = 3
    calib_split_ratio: float = 0.5
    alpha: float = 0.70
    beta: float = 0.30
    gamma: float = 0.00          # Phase 2: anchor geometry 权重
    hidden_dim: int = 64         # CoralOrdinalCalibrator MLP 宽度
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 16
    lambda_qwk: float = 0.25
    evidence_mode: str = "compact_manual"
    neural_evidence: Dict[str, object] = field(default_factory=dict)
    use_compact_evidence: bool = True
    early_reject_enabled: bool = False
    early_reject_mini_set_size: int = 16
    score_collapse_threshold: float = 0.70
    anchor_sep_threshold: float = 0.05
    early_qwk_threshold: float = 0.10
    cost_penalty_lambda: float = 0.0
    evidence_diagnostics_top_n: int = 5
    use_enhanced_evidence: bool = True
    evidence_include_raw_score: bool = True
    anchor_softmax_temperature: float = 0.10
    overfit_guard_enabled: bool = True
    overfit_gap_threshold: float = 0.15
    overfit_penalty_weight: float = 0.20
    distribution_guard_enabled: bool = True
    distribution_tv_threshold: float = 0.25
    distribution_penalty_weight: float = 0.20
    collapse_penalty_weight: float = 0.15
    max_score_overprediction_threshold: float = 3.0
    max_score_penalty_weight: float = 0.03
    diagnostic_only_skip_calibrator: bool = False
    diagnostic_sample_size: int = 0
    evidence_blocks_enabled: Dict[str, bool] = field(
        default_factory=lambda: {
            "raw": True,
            "anchor": True,
            "reasoning": True,
            "objective": True,
            "uncertainty": True,
            "stability": True,
            "enhanced_extra": True,
        }
    )

    @classmethod
    def from_config(cls, cfg: dict) -> "PaceFitnessConfig":
        p = cfg.get("pace", {})
        return cls(
            top_k_pace=p.get("top_k_pace", 3),
            calib_split_ratio=p.get("calib_split_ratio", 0.5),
            alpha=p.get("alpha", 0.70),
            beta=p.get("beta", 0.30),
            gamma=p.get("gamma", 0.00),
            hidden_dim=p.get("hidden_dim", 64),
            epochs=p.get("epochs", 20),
            lr=p.get("lr", 1e-3),
            batch_size=p.get("batch_size", 16),
            lambda_qwk=p.get("lambda_qwk", 0.25),
            evidence_mode=p.get("evidence_mode", "compact_manual"),
            neural_evidence=dict(p.get("neural_evidence", {}) or {}),
            use_compact_evidence=p.get("use_compact_evidence", True),
            early_reject_enabled=p.get("early_reject_enabled", False),
            early_reject_mini_set_size=p.get("early_reject_mini_set_size", 16),
            score_collapse_threshold=p.get("score_collapse_threshold", 0.70),
            anchor_sep_threshold=p.get("anchor_sep_threshold", 0.05),
            early_qwk_threshold=p.get("early_qwk_threshold", 0.10),
            cost_penalty_lambda=p.get("cost_penalty_lambda", 0.0),
            evidence_diagnostics_top_n=p.get("evidence_diagnostics_top_n", 5),
            use_enhanced_evidence=p.get("use_enhanced_evidence", True),
            evidence_include_raw_score=p.get("evidence_include_raw_score", True),
            anchor_softmax_temperature=p.get("anchor_softmax_temperature", 0.10),
            overfit_guard_enabled=p.get("overfit_guard_enabled", True),
            overfit_gap_threshold=p.get("overfit_gap_threshold", 0.15),
            overfit_penalty_weight=p.get("overfit_penalty_weight", 0.20),
            distribution_guard_enabled=p.get("distribution_guard_enabled", True),
            distribution_tv_threshold=p.get("distribution_tv_threshold", 0.25),
            distribution_penalty_weight=p.get("distribution_penalty_weight", 0.20),
            collapse_penalty_weight=p.get("collapse_penalty_weight", 0.15),
            max_score_overprediction_threshold=p.get("max_score_overprediction_threshold", 3.0),
            max_score_penalty_weight=p.get("max_score_penalty_weight", 0.03),
            diagnostic_only_skip_calibrator=p.get("diagnostic_only_skip_calibrator", False),
            diagnostic_sample_size=p.get("diagnostic_sample_size", 0),
            evidence_blocks_enabled={
                **cls().evidence_blocks_enabled,
                **dict(p.get("evidence_blocks_enabled", {}) or {}),
            },
        )


class ErrorType(str, Enum):
    OVER_SCORE_LOW_HIDDEN = "over_score_low_hidden"
    UNDER_SCORE_HIGH_HIDDEN = "under_score_high_hidden"
    ANCHOR_CONFUSION = "anchor_confusion"
    REASONING_CONTRADICTION = "reasoning_score_contradiction"
    RAW_COLLAPSE = "raw_collapse"
    BOUNDARY_AMBIGUITY = "boundary_ambiguity"
    GENERAL_ERROR = "general_error"


# ---------------------------------------------------------------------------
# 主评估器
# ---------------------------------------------------------------------------


class PaceFitnessEvaluator:
    """在进化循环内部为候选协议计算 PACE-guided 组合适应度。

    设计原则：
    - 无状态：每次 compute_pace_fitness 调用独立训练 calibrator，不跨代保留模型
    - 轻量：enhanced compact evidence，20 epochs，纯 CPU calibrator 训练
    - 防护：calib 样本太少或全部预测同分时优雅回退到 raw QWK
    """

    def __init__(
        self,
        local_backend: LocalLlamaBackend,
        config: PaceFitnessConfig,
        score_min: int,
        score_max: int,
    ) -> None:
        self.backend = local_backend
        self.config = config
        self.score_min = score_min
        self.score_max = score_max
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evidence_mode = str(config.evidence_mode or "compact_manual")
        hidden_input_dim = int(
            config.neural_evidence.get(
                "hidden_input_dim",
                getattr(local_backend, "hidden_dim", 4096),
            )
        )
        neural_kwargs = dict(config.neural_evidence or {})
        neural_kwargs["hidden_input_dim"] = hidden_input_dim
        self.neural_config = NeuralEvidenceConfig(**neural_kwargs)
        self.neural_encoder: Optional[NeuralEvidenceEncoder] = None
        if self.evidence_mode == "neural_hidden":
            self.neural_encoder = NeuralEvidenceEncoder(self.neural_config).to(self._device)
            self.neural_encoder.eval()
        self._last_neural_attention_stats: Dict[str, float] = {}

    def evidence_schema(self, n_anchors: int = 3) -> List[str]:
        """Return the ordered feature names used by _build_evidence_bundle."""
        blocks = self.evidence_blocks(n_anchors)
        names: List[str] = []
        for block_names in blocks.values():
            names.extend(block_names)
        return names

    def evidence_blocks(self, n_anchors: int = 3) -> Dict[str, List[str]]:
        """Return ordered evidence feature names grouped by semantic block."""
        if self.evidence_mode == "neural_hidden":
            dim = int(self.neural_config.output_dim)
            p = max(1, dim // 4)
            a = max(1, dim // 4)
            r = max(1, dim // 4)
            used = p + a + r
            s = max(0, dim - used)
            return {
                "neural_target": [f"neural_target_{i}" for i in range(p)],
                "neural_anchor_context": [f"neural_anchor_context_{i}" for i in range(a)],
                "neural_reasoning": [f"neural_reasoning_{i}" for i in range(r)],
                "neural_score": [f"neural_score_{i}" for i in range(s)],
            }
        roles = self._anchor_role_names(n_anchors)
        reasoning = [
            "reasoning_log_words",
            "reasoning_has_final_score",
            "reasoning_trait_hit_rate",
            "reasoning_strength_hit_rate",
            "reasoning_risk_hit_rate",
            "reasoning_compared_rate",
            "reasoning_example_rate",
            "reasoning_anchor_rate",
            "reasoning_score_mentions",
            "reasoning_score_span",
            "reasoning_score_std",
        ]
        objective = [
            "essay_log_words",
            "essay_log_sentences",
            "essay_log_paragraphs",
            "essay_mean_sentence_words",
            "essay_std_sentence_words",
            "essay_mean_paragraph_words",
            "essay_type_token_ratio",
            "essay_long_word_rate",
            "essay_mean_word_length",
            "essay_punctuation_rate",
            "essay_digit_rate",
        ]
        uncertainty = [
            "uncertainty_hedge_rate",
            "uncertainty_question_rate",
            "uncertainty_score_mentions",
            "uncertainty_score_mention_std",
            "uncertainty_score_first_last_delta",
            "uncertainty_raw_norm",
        ]
        raw = [
            "y_raw_norm" if self.config.evidence_include_raw_score else "y_raw_norm_disabled_zero",
            "y_raw_centered" if self.config.evidence_include_raw_score else "y_raw_centered_disabled_zero",
            "y_raw_band_low" if self.config.evidence_include_raw_score else "y_raw_band_low_disabled_zero",
            "y_raw_band_mid" if self.config.evidence_include_raw_score else "y_raw_band_mid_disabled_zero",
            "y_raw_band_high" if self.config.evidence_include_raw_score else "y_raw_band_high_disabled_zero",
        ]
        if not self.config.use_enhanced_evidence:
            anchor = (
                [f"anchor_cos_{role}" for role in roles]
                + [f"anchor_l2_raw_{role}" for role in roles]
            )
            return {
                "raw": ["y_raw_norm"],
                "anchor": anchor,
                "reasoning": reasoning,
                "objective": objective,
                "uncertainty": uncertainty,
                "stability": [],
                "enhanced_extra": [],
            }

        anchor = (
            [f"anchor_cos_{role}" for role in roles]
            + [f"anchor_cos_dist_{role}" for role in roles]
            + [f"anchor_l2_norm_{role}" for role in roles]
            + [f"anchor_prob_{role}" for role in roles]
            + [f"anchor_closest_{role}" for role in roles]
            + [
                "anchor_last_first_cos_delta",
                "anchor_top2_cos_margin",
                "anchor_expected_index_norm",
                "anchor_raw_expected_gap",
                "anchor_nearest_index_norm",
                "anchor_raw_nearest_gap",
                "anchor_boundary_pair_prob_mass",
                "anchor_boundary_cos_margin",
                "anchor_hidden_raw_band_agreement",
            ]
        )
        return {
            "raw": raw,
            "anchor": anchor,
            "reasoning": reasoning,
            "objective": objective,
            "uncertainty": uncertainty,
            "stability": [],
            "enhanced_extra": [],
        }

    def evidence_dim(self, n_anchors: int = 3) -> int:
        return len(self.evidence_schema(n_anchors))

    def evidence_block_slices(self, n_anchors: int = 3) -> Dict[str, List[int]]:
        blocks = self.evidence_blocks(n_anchors)
        mapping: Dict[str, List[int]] = {}
        offset = 0
        for name, names in blocks.items():
            mapping[name] = list(range(offset, offset + len(names)))
            offset += len(names)
        return mapping

    def evidence_block_dims(self, n_anchors: int = 3) -> Dict[str, int]:
        return {k: len(v) for k, v in self.evidence_block_slices(n_anchors).items()}

    def enabled_evidence_blocks(self) -> Dict[str, bool]:
        if self.evidence_mode == "neural_hidden":
            return {
                "neural_target": True,
                "neural_anchor_context": self.neural_config.anchor_view != "none",
                "neural_reasoning": bool(self.neural_config.use_reasoning_hidden),
                "neural_score": bool(
                    self.neural_config.use_raw_score_embedding
                    or self.neural_config.use_score_distribution
                ),
            }
        defaults = {
            "raw": True,
            "anchor": True,
            "reasoning": True,
            "objective": True,
            "uncertainty": True,
            "stability": True,
            "enhanced_extra": True,
        }
        defaults.update(dict(self.config.evidence_blocks_enabled or {}))
        return defaults

    def evidence_metadata(self, n_anchors: int = 3) -> Dict[str, object]:
        payload = {
            "evidence_dim": self.evidence_dim(n_anchors),
            "evidence_schema": self.evidence_schema(n_anchors),
            "evidence_block_slices": self.evidence_block_slices(n_anchors),
            "enabled_blocks": self.enabled_evidence_blocks(),
            "block_dims": self.evidence_block_dims(n_anchors),
            "evidence_mode": self.evidence_mode,
        }
        if self.evidence_mode == "neural_hidden":
            payload.update({
                "neural_anchor_view": self.neural_config.anchor_view,
                "neural_output_dim": self.neural_config.output_dim,
                "neural_attention_mode": self.neural_config.attention_mode,
                "neural_projection_seed": self.neural_config.projection_seed,
                "neural_preserve_block_structure": self.neural_config.preserve_block_structure,
                "neural_attention_stats": dict(self._last_neural_attention_stats),
            })
        return payload

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def score_essays(
        self,
        items: List[Dict],
        instruction: str,
        static_exemplars: List[Dict],
    ) -> List[ScoringResult]:
        """用本地模型逐篇评分，返回 y_raw + scoring-context hidden state。"""
        results = []
        for item in items:
            req = ScoringRequest(
                essay_id=item["essay_id"],
                essay_text=item["essay_text"],
                instruction=instruction,
                static_exemplars=static_exemplars,
                score_min=self.score_min,
                score_max=self.score_max,
                dynamic_ex="(None)",  # PACE 模式下关闭 RAG，保持 prompt 确定性
            )
            result = self.backend.score(req)
            result.hidden = self.backend.encode_scoring_context(
                instruction=instruction,
                static_exemplars=static_exemplars,
                essay_text=item["essay_text"],
                score_min=self.score_min,
                score_max=self.score_max,
                representation_target="Encode the essay to be scored.",
            )
            results.append(result)
        return results

    def compute_anchor_hiddens(
        self,
        static_exemplars: List[Dict],
        instruction: str,
        return_dual: bool = False,
    ) -> torch.Tensor | Dict[str, object]:
        """计算 anchor essay 的 scoring-context hidden states。

        返回 shape (n_anchors, hidden_dim)。
        """
        hiddens_score = []
        hiddens_text = []
        roles = self._anchor_role_names(len(static_exemplars))
        for idx, ex in enumerate(static_exemplars):
            role = roles[idx] if idx < len(roles) else self._score_band_label(ex["domain1_score"])
            if return_dual:
                text_hidden = self.backend.encode_scoring_context(
                    instruction=instruction,
                    static_exemplars=static_exemplars,
                    essay_text=ex["essay_text"],
                    score_min=self.score_min,
                    score_max=self.score_max,
                    known_score=None,
                    representation_target="Encode this essay as a reference essay without using its score label.",
                )
                hiddens_text.append(text_hidden.float())
            hidden = self.backend.encode_scoring_context(
                instruction=instruction,
                static_exemplars=static_exemplars,
                essay_text=ex["essay_text"],
                score_min=self.score_min,
                score_max=self.score_max,
                known_score=ex["domain1_score"],
                representation_target=f"Encode this reference essay as a {role}-score anchor.",
            )
            hiddens_score.append(hidden.float())
        score_stack = torch.stack(hiddens_score, dim=0)
        if not return_dual:
            return score_stack  # (n_anchors, hidden_dim)
        text_stack = torch.stack(hiddens_text, dim=0) if hiddens_text else score_stack
        return {
            "anchor_hiddens_text": text_stack,
            "anchor_hiddens_score": score_stack,
            "anchor_scores": [int(ex["domain1_score"]) for ex in static_exemplars],
        }

    def _anchor_forward_pass_count(self, static_exemplars: List[Dict]) -> int:
        multiplier = 2 if self.evidence_mode == "neural_hidden" else 1
        return int(len(static_exemplars) * multiplier)

    def _compute_diagnostic_only_probe(
        self,
        *,
        items: List[Dict],
        results: List[ScoringResult],
        anchor_hiddens: torch.Tensor,
        anchor_scores: List[int],
        evidence_schema: List[str],
        evidence_payload: Dict[str, object],
        anchor_sec: float,
        fitness_sec: float,
        usage_before: Dict[str, int],
        static_exemplars: List[Dict],
    ) -> Dict:
        """Return hidden-error diagnostics without training a PACE calibrator."""
        y_true = [int(item["domain1_score"]) for item in items]
        y_raw = [int(result.y_raw) for result in results]
        essay_hiddens = [result.hidden.float() for result in results if result.hidden is not None]
        if essay_hiddens:
            anchor_geometry = self.compute_anchor_geometry(
                anchor_hiddens=anchor_hiddens,
                essay_hiddens=torch.stack(essay_hiddens, dim=0),
                y_true=y_true,
                anchor_scores=anchor_scores,
            )
        else:
            anchor_geometry = {
                "anchor_geometry_score": 0.0,
                "anchor_separation": 0.0,
                "anchor_separation_raw": 0.0,
                "anchor_ordinal_consistency": 0.0,
                "anchor_monotonicity": 0.0,
                "anchor_min_pair": [],
            }
        diagnostics = self.classify_error_types(
            items=items,
            results=results,
            y_true=y_true,
            anchor_hiddens=anchor_hiddens,
            anchor_scores=anchor_scores,
            anchor_geometry=anchor_geometry,
        )
        distribution_metrics = self._prediction_distribution_metrics(y_true, y_raw)
        usage_delta = self.backend.usage_delta(usage_before)
        total_sec = anchor_sec + fitness_sec
        anchor_passes = self._anchor_forward_pass_count(static_exemplars)
        return {
            "pace_qwk": 0.0,
            "raw_qwk": self._safe_qwk(y_true, y_raw),
            "calib_pace_qwk": None,
            "calib_raw_qwk": None,
            "overfit_gap": 0.0,
            "overfit_penalty": 0.0,
            "distribution_penalty": 0.0,
            "pace_distribution_metrics": distribution_metrics,
            **self._distribution_flat_payload(distribution_metrics),
            "anchor_geometry_score": anchor_geometry["anchor_geometry_score"],
            "anchor_separation": anchor_geometry["anchor_separation"],
            "anchor_separation_raw": anchor_geometry["anchor_separation_raw"],
            "anchor_ordinal_consistency": anchor_geometry["anchor_ordinal_consistency"],
            "anchor_monotonicity": anchor_geometry["anchor_monotonicity"],
            "anchor_min_pair": anchor_geometry["anchor_min_pair"],
            "pace_diagnostics": diagnostics["diagnostics"],
            "pace_diagnostic_summary": diagnostics["summary"],
            "dominant_error_type": diagnostics["dominant_error_type"],
            "suggested_anchor_mutation_slot": diagnostics["suggested_anchor_mutation_slot"],
            "early_rejection_metrics": {},
            "cost_penalty": 0.0,
            "local_usage": usage_delta,
            "evidence_dim": len(evidence_schema),
            "evidence_schema": evidence_schema,
            **evidence_payload,
            "calibrator_probe_combined": 0.0,
            "pace_qwk_used_for_selection": False,
            "diagnostic_only_skip_calibrator": True,
            "anchor_inference_sec": round(anchor_sec, 2),
            "calib_inference_sec": 0.0,
            "fitness_inference_sec": round(fitness_sec, 2),
            "calibrator_train_sec": 0.0,
            "total_pace_sec": round(total_sec, 2),
            "n_calib": 0,
            "n_fitness": len(results),
            "scoring_forward_passes": len(results),
            "representation_forward_passes": anchor_passes + len(results),
            "total_local_forward_passes": anchor_passes + 2 * len(results),
            "local_prompt_tokens": usage_delta.get("prompt_tokens", 0),
            "local_completion_tokens": usage_delta.get("completion_tokens", 0),
            "local_representation_tokens": usage_delta.get("representation_tokens", 0),
        }

    def compute_pace_fitness(
        self,
        protocol: "object",   # PromptIndividual，用 object 避免循环导入
        calib_items: List[Dict],
        fitness_items: List[Dict],
    ) -> Dict:
        """计算一个协议的 PACE 组合适应度。

        返回字典包含：
          pace_qwk, raw_qwk, calibrator_probe_combined,
          anchor_inference_sec, calib_inference_sec,
          fitness_inference_sec, calibrator_train_sec
        """
        instruction = (
            protocol.scoring_instruction_text()
            if hasattr(protocol, "scoring_instruction_text")
            else protocol.instruction_text
        )  # type: ignore[attr-defined]
        static_exemplars = protocol.static_exemplars  # type: ignore[attr-defined]
        usage_before = self.backend.usage_snapshot()

        # 1. Anchor hidden states
        t0 = time.time()
        try:
            anchor_payload = None
            if self.evidence_mode == "neural_hidden":
                anchor_payload = self.compute_anchor_hiddens(
                    static_exemplars,
                    instruction,
                    return_dual=True,
                )
                anchor_hiddens = anchor_payload["anchor_hiddens_score"]  # type: ignore[index]
            else:
                anchor_hiddens = self.compute_anchor_hiddens(static_exemplars, instruction)
        except Exception as e:
            print(f"    [PACE] anchor_hiddens failed: {e}. Returning raw QWK as fallback.")
            return self._fallback_result(extra=self._usage_payload(usage_before))
        anchor_sec = time.time() - t0
        evidence_schema = self.evidence_schema(anchor_hiddens.shape[0])
        evidence_payload = self.evidence_metadata(anchor_hiddens.shape[0])

        if self.config.diagnostic_only_skip_calibrator:
            diag_items = list(fitness_items)
            n_diag = int(self.config.diagnostic_sample_size or 0)
            if n_diag > 0 and n_diag < len(diag_items):
                diag_items = self._select_mini_items(diag_items, n_diag)
            t0 = time.time()
            diag_results = self.score_essays(diag_items, instruction, static_exemplars)
            diag_sec = time.time() - t0
            return self._compute_diagnostic_only_probe(
                items=diag_items,
                results=diag_results,
                anchor_hiddens=anchor_hiddens,
                anchor_scores=[ex["domain1_score"] for ex in static_exemplars],
                evidence_schema=evidence_schema,
                evidence_payload=evidence_payload,
                anchor_sec=anchor_sec,
                fitness_sec=diag_sec,
                usage_before=usage_before,
                static_exemplars=static_exemplars,
            )

        early_sec = 0.0
        early_results: Optional[List[ScoringResult]] = None
        early_result_by_id: Dict[object, ScoringResult] = {}
        early_rejection = {
            "rejected": False,
            "reason": "",
            "metrics": {},
        }
        if self.config.early_reject_enabled and fitness_items:
            t0 = time.time()
            n_mini = min(self.config.early_reject_mini_set_size, len(fitness_items))
            mini_items = self._select_mini_items(fitness_items, n_mini)
            early_results = self.score_essays(mini_items, instruction, static_exemplars)
            early_result_by_id = {
                item["essay_id"]: result for item, result in zip(mini_items, early_results)
            }
            early_sec = time.time() - t0
            early_rejection = self._early_rejection_check(
                items=mini_items,
                results=early_results,
                anchor_hiddens=anchor_hiddens,
            )
            if early_rejection["rejected"]:
                print(
                    "    [PACE] early rejected: "
                    f"{early_rejection['reason']} "
                    f"metrics={early_rejection['metrics']}"
                )
                return self._fallback_result(
                    raw_qwk=float(early_rejection["metrics"].get("early_qwk", 0.0)),
                    extra={
                        "_early_rejected": True,
                        "early_rejection_reason": early_rejection["reason"],
                        "early_rejection_metrics": early_rejection["metrics"],
                        "fitness_inference_sec": round(early_sec, 2),
                        "total_pace_sec": round(anchor_sec + early_sec, 2),
                        "n_fitness": len(early_results),
                        "evidence_dim": len(evidence_schema),
                        "evidence_schema": evidence_schema,
                        **evidence_payload,
                        "scoring_forward_passes": len(early_results),
                        "representation_forward_passes": len(static_exemplars) + len(early_results),
                        "total_local_forward_passes": len(static_exemplars) + 2 * len(early_results),
                        **self._usage_payload(usage_before),
                    },
                )

        # 2. Score calib essays
        t0 = time.time()
        calib_results = self.score_essays(calib_items, instruction, static_exemplars)
        calib_sec = time.time() - t0

        # 3. Build evidence vectors for calib set
        z_calib_list = []
        y_calib_list = []
        y_calib_raw_list = []
        for item, result in zip(calib_items, calib_results):
            if result.hidden is None:
                continue
            z = self._build_evidence_bundle(
                result,
                item["essay_text"],
                anchor_hiddens,
                anchor_scores=[ex["domain1_score"] for ex in static_exemplars],
                anchor_payload=anchor_payload,
            )
            z_calib_list.append(z)
            y_calib_list.append(item["domain1_score"])
            y_calib_raw_list.append(result.y_raw)

        if len(z_calib_list) < 4 or len(set(y_calib_list)) < 2:
            print(
                f"    [PACE] calib set too small or uniform (n={len(z_calib_list)}, "
                f"unique_y={len(set(y_calib_list))}). Falling back to raw QWK."
            )
            return self._fallback_result(
                extra={
                    "evidence_dim": len(evidence_schema),
                    "evidence_schema": evidence_schema,
                    **evidence_payload,
                    "anchor_inference_sec": round(anchor_sec, 2),
                    "calib_inference_sec": round(calib_sec, 2),
                    "total_pace_sec": round(anchor_sec + calib_sec, 2),
                    "n_calib": len(z_calib_list),
                    "scoring_forward_passes": len(calib_items),
                    "representation_forward_passes": len(static_exemplars) + len(calib_items),
                    "total_local_forward_passes": len(static_exemplars) + 2 * len(calib_items),
                    **self._usage_payload(usage_before),
                }
            )

        z_calib = torch.stack(z_calib_list, dim=0)
        y_calib = torch.tensor(y_calib_list, dtype=torch.long)

        # 4. Train lightweight calibrator on calib set
        t0 = time.time()
        calibrator = self._train_calibrator(z_calib, y_calib)
        train_sec = time.time() - t0

        # 5. Score fitness essays
        t0 = time.time()
        if early_results is not None:
            remaining_items = [
                item for item in fitness_items if item["essay_id"] not in early_result_by_id
            ]
            remaining_results = self.score_essays(remaining_items, instruction, static_exemplars)
            result_by_id = dict(early_result_by_id)
            result_by_id.update(
                {item["essay_id"]: result for item, result in zip(remaining_items, remaining_results)}
            )
            fitness_results = [result_by_id[item["essay_id"]] for item in fitness_items]
            fitness_sec = early_sec + (time.time() - t0)
        else:
            fitness_results = self.score_essays(fitness_items, instruction, static_exemplars)
            fitness_sec = time.time() - t0

        # 6. Build evidence vectors for fitness set
        z_fitness_list = []
        y_fitness_true = []
        y_fitness_raw = []
        for item, result in zip(fitness_items, fitness_results):
            if result.hidden is None:
                continue
            z = self._build_evidence_bundle(
                result,
                item["essay_text"],
                anchor_hiddens,
                anchor_scores=[ex["domain1_score"] for ex in static_exemplars],
                anchor_payload=anchor_payload,
            )
            z_fitness_list.append(z)
            y_fitness_true.append(item["domain1_score"])
            y_fitness_raw.append(result.y_raw)

        if len(z_fitness_list) < 4:
            print(f"    [PACE] fitness set too small (n={len(z_fitness_list)}). Falling back.")
            return self._fallback_result(
                extra={
                    "evidence_dim": len(evidence_schema),
                    "evidence_schema": evidence_schema,
                    **evidence_payload,
                    "anchor_inference_sec": round(anchor_sec, 2),
                    "calib_inference_sec": round(calib_sec, 2),
                    "fitness_inference_sec": round(fitness_sec, 2),
                    "calibrator_train_sec": round(train_sec, 2),
                    "total_pace_sec": round(anchor_sec + calib_sec + fitness_sec + train_sec, 2),
                    "n_calib": len(z_calib_list),
                    "n_fitness": len(z_fitness_list),
                    "scoring_forward_passes": len(calib_items) + len(fitness_items),
                    "representation_forward_passes": len(static_exemplars) + len(calib_items) + len(fitness_items),
                    "total_local_forward_passes": len(static_exemplars) + 2 * (len(calib_items) + len(fitness_items)),
                    **self._usage_payload(usage_before),
                }
            )

        z_fitness = torch.stack(z_fitness_list, dim=0)

        # 7. PACE predictions on fitness set
        calibrator.eval()
        with torch.no_grad():
            pace_preds_tensor = calibrator.predict_scores(
                z_fitness.to(self._device), decode_mode="threshold"
            )
            calib_preds_tensor = calibrator.predict_scores(
                z_calib.to(self._device), decode_mode="threshold"
            )
        pace_preds = pace_preds_tensor.cpu().numpy().tolist()
        calib_preds = calib_preds_tensor.cpu().numpy().tolist()

        # 8. Compute metrics
        pace_qwk = self._safe_qwk(y_fitness_true, pace_preds)
        raw_qwk = self._safe_qwk(y_fitness_true, y_fitness_raw)
        calib_pace_qwk = self._safe_qwk(y_calib_list, calib_preds)
        calib_raw_qwk = self._safe_qwk(y_calib_list, y_calib_raw_list)
        overfit_gap = max(0.0, calib_pace_qwk - pace_qwk)
        overfit_penalty = 0.0
        if self.config.overfit_guard_enabled:
            excess_gap = max(0.0, overfit_gap - self.config.overfit_gap_threshold)
            overfit_penalty = self.config.overfit_penalty_weight * excess_gap
        distribution_metrics = self._prediction_distribution_metrics(
            y_fitness_true, pace_preds
        )
        distribution_penalty = self._distribution_penalty(distribution_metrics)

        anchor_geometry = self.compute_anchor_geometry(
            anchor_hiddens=anchor_hiddens,
            essay_hiddens=torch.stack([z.hidden.float() for z in fitness_results if z.hidden is not None], dim=0),
            y_true=y_fitness_true,
            anchor_scores=[ex["domain1_score"] for ex in static_exemplars],
        )
        diagnostics = self.classify_error_types(
            items=fitness_items,
            results=fitness_results,
            y_true=y_fitness_true,
            anchor_hiddens=anchor_hiddens,
            anchor_scores=[ex["domain1_score"] for ex in static_exemplars],
            anchor_geometry=anchor_geometry,
        )
        total_sec = anchor_sec + calib_sec + fitness_sec + train_sec
        cost_penalty = self.config.cost_penalty_lambda * self._cost_scale(total_sec)
        usage_delta = self.backend.usage_delta(usage_before)
        evidence_payload = self.evidence_metadata(anchor_hiddens.shape[0])
        calibrator_probe_combined = (
            self.config.alpha * pace_qwk
            + self.config.beta * raw_qwk
            + self.config.gamma * anchor_geometry["anchor_geometry_score"]
            - cost_penalty
            - overfit_penalty
            - distribution_penalty
        )
        if not np.isfinite(calibrator_probe_combined):
            print("    [PACE] calibrator probe score is non-finite. Falling back.")
            return self._fallback_result(
                extra={
                    "evidence_dim": len(evidence_schema),
                    "evidence_schema": evidence_schema,
                    **evidence_payload,
                    **self._usage_payload(usage_before),
                }
            )

        return {
            "pace_qwk": pace_qwk,
            "raw_qwk": raw_qwk,
            "calib_pace_qwk": calib_pace_qwk,
            "calib_raw_qwk": calib_raw_qwk,
            "overfit_gap": overfit_gap,
            "overfit_penalty": overfit_penalty,
            "distribution_penalty": distribution_penalty,
            "pace_distribution_metrics": distribution_metrics,
            **self._distribution_flat_payload(distribution_metrics),
            "anchor_geometry_score": anchor_geometry["anchor_geometry_score"],
            "anchor_separation": anchor_geometry["anchor_separation"],
            "anchor_separation_raw": anchor_geometry["anchor_separation_raw"],
            "anchor_ordinal_consistency": anchor_geometry["anchor_ordinal_consistency"],
            "anchor_monotonicity": anchor_geometry["anchor_monotonicity"],
            "anchor_min_pair": anchor_geometry["anchor_min_pair"],
            "pace_diagnostics": diagnostics["diagnostics"],
            "pace_diagnostic_summary": diagnostics["summary"],
            "dominant_error_type": diagnostics["dominant_error_type"],
            "suggested_anchor_mutation_slot": diagnostics["suggested_anchor_mutation_slot"],
            "early_rejection_metrics": early_rejection["metrics"],
            "cost_penalty": cost_penalty,
            "local_usage": usage_delta,
            "evidence_dim": len(evidence_schema),
            "evidence_schema": evidence_schema,
            **evidence_payload,
            "calibrator_probe_combined": calibrator_probe_combined,
            "pace_qwk_used_for_selection": False,
            "anchor_inference_sec": round(anchor_sec, 2),
            "calib_inference_sec": round(calib_sec, 2),
            "fitness_inference_sec": round(fitness_sec, 2),
            "calibrator_train_sec": round(train_sec, 2),
            "total_pace_sec": round(total_sec, 2),
            "n_calib": len(z_calib_list),
            "n_fitness": len(z_fitness_list),
            "scoring_forward_passes": len(calib_items) + len(fitness_items),
            "representation_forward_passes": len(static_exemplars) + len(calib_items) + len(fitness_items),
            "total_local_forward_passes": len(static_exemplars) + 2 * (len(calib_items) + len(fitness_items)),
            "local_prompt_tokens": usage_delta.get("prompt_tokens", 0),
            "local_completion_tokens": usage_delta.get("completion_tokens", 0),
            "local_representation_tokens": usage_delta.get("representation_tokens", 0),
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def score_with_pace_calibrator(
        self,
        protocol: "object",
        calib_items: List[Dict],
        eval_items: List[Dict],
    ) -> Dict:
        """Train PACE calibrator on calib_items and evaluate it on eval_items."""
        instruction = (
            protocol.scoring_instruction_text()
            if hasattr(protocol, "scoring_instruction_text")
            else protocol.instruction_text
        )  # type: ignore[attr-defined]
        static_exemplars = protocol.static_exemplars  # type: ignore[attr-defined]
        usage_before = self.backend.usage_snapshot()
        total_start = time.time()

        try:
            t0 = time.time()
            anchor_payload = None
            if self.evidence_mode == "neural_hidden":
                anchor_payload = self.compute_anchor_hiddens(
                    static_exemplars,
                    instruction,
                    return_dual=True,
                )
                anchor_hiddens = anchor_payload["anchor_hiddens_score"]  # type: ignore[index]
            else:
                anchor_hiddens = self.compute_anchor_hiddens(static_exemplars, instruction)
            anchor_sec = time.time() - t0
            evidence_schema = self.evidence_schema(anchor_hiddens.shape[0])
            evidence_payload = self.evidence_metadata(anchor_hiddens.shape[0])
        except Exception as e:
            print(f"    [PACE Final] anchor_hiddens failed: {e}.")
            return self._fallback_result(extra=self._usage_payload(usage_before))

        t0 = time.time()
        calib_results = self.score_essays(calib_items, instruction, static_exemplars)
        calib_sec = time.time() - t0

        z_calib_list = []
        y_calib_list = []
        y_calib_raw_list = []
        for item, result in zip(calib_items, calib_results):
            if result.hidden is None:
                continue
            z_calib_list.append(
                self._build_evidence_bundle(
                    result,
                    item["essay_text"],
                    anchor_hiddens,
                    anchor_scores=[ex["domain1_score"] for ex in static_exemplars],
                    anchor_payload=anchor_payload,
                )
            )
            y_calib_list.append(item["domain1_score"])
            y_calib_raw_list.append(result.y_raw)

        if len(z_calib_list) < 4 or len(set(y_calib_list)) < 2:
            return self._fallback_result(
                extra={
                    "_fallback_reason": "calib_too_small_or_uniform",
                    "evidence_dim": len(evidence_schema),
                    "evidence_schema": evidence_schema,
                    **evidence_payload,
                    "anchor_inference_sec": round(anchor_sec, 2),
                    "calib_inference_sec": round(calib_sec, 2),
                    "total_pace_sec": round(time.time() - total_start, 2),
                    "n_calib": len(z_calib_list),
                    "n_eval": 0,
                    **self._usage_payload(usage_before),
                }
            )

        z_calib = torch.stack(z_calib_list, dim=0)
        y_calib = torch.tensor(y_calib_list, dtype=torch.long)

        t0 = time.time()
        calibrator = self._train_calibrator(z_calib, y_calib)
        train_sec = time.time() - t0

        t0 = time.time()
        eval_results = self.score_essays(eval_items, instruction, static_exemplars)
        eval_sec = time.time() - t0

        z_eval_list = []
        y_eval_true = []
        y_eval_raw = []
        eval_ids = []
        for item, result in zip(eval_items, eval_results):
            if result.hidden is None:
                continue
            z_eval_list.append(
                self._build_evidence_bundle(
                    result,
                    item["essay_text"],
                    anchor_hiddens,
                    anchor_scores=[ex["domain1_score"] for ex in static_exemplars],
                    anchor_payload=anchor_payload,
                )
            )
            y_eval_true.append(item["domain1_score"])
            y_eval_raw.append(result.y_raw)
            eval_ids.append(item.get("essay_id"))

        if len(z_eval_list) < 4:
            raw_qwk = self._safe_qwk(y_eval_true, y_eval_raw) if y_eval_true else 0.0
            return self._fallback_result(
                raw_qwk=raw_qwk,
                extra={
                    "_fallback_reason": "eval_too_small",
                    "evidence_dim": len(evidence_schema),
                    "evidence_schema": evidence_schema,
                    **evidence_payload,
                    "anchor_inference_sec": round(anchor_sec, 2),
                    "calib_inference_sec": round(calib_sec, 2),
                    "fitness_inference_sec": round(eval_sec, 2),
                    "calibrator_train_sec": round(train_sec, 2),
                    "total_pace_sec": round(time.time() - total_start, 2),
                    "n_calib": len(z_calib_list),
                    "n_eval": len(z_eval_list),
                    "y_true": y_eval_true,
                    "y_raw": y_eval_raw,
                    "essay_ids": eval_ids,
                    **self._usage_payload(usage_before),
                }
            )

        z_eval = torch.stack(z_eval_list, dim=0)
        calibrator.eval()
        with torch.no_grad():
            pace_preds_tensor = calibrator.predict_scores(
                z_eval.to(self._device), decode_mode="threshold"
            )
            calib_preds_tensor = calibrator.predict_scores(
                z_calib.to(self._device), decode_mode="threshold"
            )
        pace_preds = pace_preds_tensor.cpu().numpy().tolist()
        calib_preds = calib_preds_tensor.cpu().numpy().tolist()

        pace_qwk = self._safe_qwk(y_eval_true, pace_preds)
        raw_qwk = self._safe_qwk(y_eval_true, y_eval_raw)
        calib_pace_qwk = self._safe_qwk(y_calib_list, calib_preds)
        calib_raw_qwk = self._safe_qwk(y_calib_list, y_calib_raw_list)
        overfit_gap = max(0.0, calib_pace_qwk - pace_qwk)
        distribution_metrics = self._prediction_distribution_metrics(
            y_eval_true, pace_preds
        )
        distribution_penalty = self._distribution_penalty(distribution_metrics)
        usage_delta = self.backend.usage_delta(usage_before)
        total_sec = time.time() - total_start
        evidence_payload = self.evidence_metadata(anchor_hiddens.shape[0])

        return {
            "pace_qwk": pace_qwk,
            "raw_qwk": raw_qwk,
            "calibrated_test_qwk": pace_qwk,
            "raw_test_qwk_same_pass": raw_qwk,
            "calib_pace_qwk": calib_pace_qwk,
            "calib_raw_qwk": calib_raw_qwk,
            "overfit_gap": overfit_gap,
            "distribution_penalty": distribution_penalty,
            "pace_distribution_metrics": distribution_metrics,
            **self._distribution_flat_payload(distribution_metrics),
            "n_calib": len(z_calib_list),
            "n_eval": len(z_eval_list),
            "y_true": y_eval_true,
            "y_raw": y_eval_raw,
            "y_pace": pace_preds,
            "essay_ids": eval_ids,
            "evidence_dim": len(evidence_schema),
            "evidence_schema": evidence_schema,
            **evidence_payload,
            "anchor_inference_sec": round(anchor_sec, 2),
            "calib_inference_sec": round(calib_sec, 2),
            "fitness_inference_sec": round(eval_sec, 2),
            "calibrator_train_sec": round(train_sec, 2),
            "total_pace_sec": round(total_sec, 2),
            "scoring_forward_passes": len(calib_items) + len(eval_items),
            "representation_forward_passes": len(static_exemplars) + len(calib_items) + len(eval_items),
            "total_local_forward_passes": len(static_exemplars) + 2 * (len(calib_items) + len(eval_items)),
            "local_usage": usage_delta,
            "local_prompt_tokens": usage_delta.get("prompt_tokens", 0),
            "local_completion_tokens": usage_delta.get("completion_tokens", 0),
            "local_representation_tokens": usage_delta.get("representation_tokens", 0),
            "_fallback": False,
        }

    def _build_evidence_bundle(
        self,
        result: ScoringResult,
        essay_text: str,
        anchor_hiddens: torch.Tensor,
        anchor_scores: Optional[List[int]] = None,
        anchor_payload: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        """Build compact, schema-backed evidence vector z.

        Enhanced mode (default, 3 anchors) produces 52 dimensions:
        raw score features (5), anchor geometry features (19), reasoning
        features (11), objective essay features (11), and uncertainty features
        (6).
        """
        if self.evidence_mode == "neural_hidden":
            return self._build_neural_evidence_bundle(
                result=result,
                anchor_hiddens=anchor_hiddens,
                anchor_scores=anchor_scores,
                anchor_payload=anchor_payload,
            )
        h = result.hidden.float()  # (hidden_dim,)
        anchors = anchor_hiddens.float()  # (n_anchors, hidden_dim)
        n_anchors = int(anchors.shape[0])

        score_span = max(1, self.score_max - self.score_min)
        y_norm_value = float(np.clip((result.y_raw - self.score_min) / score_span, 0.0, 1.0))
        if not self.config.evidence_include_raw_score:
            y_norm_feature_value = 0.0
            y_centered_feature_value = 0.0
            raw_band_enabled = False
        else:
            y_norm_feature_value = y_norm_value
            y_centered_feature_value = 2.0 * y_norm_value - 1.0
            raw_band_enabled = True
        y_norm = torch.tensor([y_norm_feature_value], dtype=torch.float32)

        cos_sims = F.cosine_similarity(h.unsqueeze(0), anchors, dim=1).cpu()
        l2_raw = (h.unsqueeze(0) - anchors).norm(p=2, dim=1).cpu()

        r_s = build_reasoning_features(result.raw_text, self.score_min, self.score_max)
        f_o = build_objective_features(essay_text)
        u = build_uncertainty_features(result.raw_text, result.y_raw, self.score_min, self.score_max)

        if not self.config.use_enhanced_evidence:
            z = torch.cat([y_norm, cos_sims, l2_raw, r_s, f_o, u], dim=0).float()
            return self._validate_evidence_vector(z, n_anchors)

        y_centered = torch.tensor([y_centered_feature_value], dtype=torch.float32)
        band = self._score_band_index(result.y_raw)
        band_one_hot = torch.zeros(3, dtype=torch.float32)
        if raw_band_enabled:
            band_one_hot[min(2, max(0, band))] = 1.0

        cos_dists = ((1.0 - cos_sims) / 2.0).clamp(0.0, 1.0)
        hidden_scale = max(1.0, math.sqrt(float(h.numel())))
        l2_norms = (l2_raw / hidden_scale).clamp(0.0, 10.0)
        tau = max(float(self.config.anchor_softmax_temperature), 1e-4)
        anchor_probs = torch.softmax(cos_sims / tau, dim=0)

        closest_one_hot = torch.zeros(n_anchors, dtype=torch.float32)
        closest_idx = int(torch.argmax(cos_sims).item()) if n_anchors else 0
        if n_anchors:
            closest_one_hot[closest_idx] = 1.0

        last_first_delta = (
            float(cos_sims[-1] - cos_sims[0]) if n_anchors >= 2 else 0.0
        )
        if n_anchors >= 2:
            top2 = torch.topk(cos_sims, k=2).values
            top2_margin = float(top2[0] - top2[1])
        else:
            top2_margin = 0.0
        if n_anchors >= 2:
            anchor_idx = torch.arange(n_anchors, dtype=torch.float32)
            expected_idx_norm = float(torch.dot(anchor_probs, anchor_idx) / (n_anchors - 1))
        else:
            expected_idx_norm = 0.0
        raw_expected_gap = y_norm_value - expected_idx_norm if self.config.evidence_include_raw_score else 0.0
        nearest_idx_norm = float(closest_idx / max(1, n_anchors - 1)) if n_anchors else 0.0
        if anchor_scores and 0 <= closest_idx < len(anchor_scores):
            raw_nearest_gap = float((result.y_raw - int(anchor_scores[closest_idx])) / score_span) if self.config.evidence_include_raw_score else 0.0
        else:
            raw_nearest_gap = 0.0
        if n_anchors >= 4:
            boundary_pair_prob_mass = float(anchor_probs[1] + anchor_probs[2])
            boundary_cos_margin = float(cos_sims[2] - cos_sims[1])
            nearest_band = 0 if closest_idx == 0 else (2 if closest_idx == n_anchors - 1 else 1)
        else:
            boundary_pair_prob_mass = float(anchor_probs[1]) if n_anchors >= 3 else 0.0
            boundary_cos_margin = 0.0
            nearest_band = min(2, max(0, closest_idx))
        hidden_raw_band_agreement = (
            1.0 if self.config.evidence_include_raw_score and int(band) == int(nearest_band) else 0.0
        )

        anchor_scalars = torch.tensor(
            [
                last_first_delta,
                top2_margin,
                expected_idx_norm,
                raw_expected_gap,
                nearest_idx_norm,
                raw_nearest_gap,
                boundary_pair_prob_mass,
                boundary_cos_margin,
                hidden_raw_band_agreement,
            ],
            dtype=torch.float32,
        )
        z = torch.cat(
            [
                y_norm,
                y_centered,
                band_one_hot,
                cos_sims,
                cos_dists,
                l2_norms,
                anchor_probs,
                closest_one_hot,
                anchor_scalars,
                r_s,
                f_o,
                u,
            ],
            dim=0,
        ).float()
        return self._validate_evidence_vector(z, n_anchors)

    def _build_neural_evidence_bundle(
        self,
        *,
        result: ScoringResult,
        anchor_hiddens: torch.Tensor,
        anchor_scores: Optional[List[int]],
        anchor_payload: Optional[Dict[str, object]],
    ) -> torch.Tensor:
        if result.hidden is None:
            raise ValueError("neural_hidden evidence requires result.hidden")
        if self.neural_encoder is None:
            self.neural_encoder = NeuralEvidenceEncoder(self.neural_config).to(self._device)
            self.neural_encoder.eval()
        target_hidden = result.hidden.float().to(self._device).unsqueeze(0)
        payload = anchor_payload or {}
        anchor_text = payload.get("anchor_hiddens_text")
        anchor_score = payload.get("anchor_hiddens_score")
        if anchor_text is None:
            anchor_text = anchor_hiddens
        if anchor_score is None:
            anchor_score = anchor_hiddens
        scores = payload.get("anchor_scores") or anchor_scores or []
        anchor_scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self._device)

        reasoning_hidden = None
        if self.neural_config.use_reasoning_hidden:
            try:
                reasoning_hidden = self.backend.encode_text_mean(result.raw_text).float().to(self._device)
            except Exception:
                reasoning_hidden = torch.zeros_like(target_hidden[0])

        y_raw = torch.tensor([result.y_raw], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            z, aux = self.neural_encoder(
                target_hidden=target_hidden,
                anchor_hidden_text=anchor_text.to(self._device) if isinstance(anchor_text, torch.Tensor) else None,
                anchor_hidden_score=anchor_score.to(self._device) if isinstance(anchor_score, torch.Tensor) else None,
                anchor_scores=anchor_scores_tensor,
                score_min=self.score_min,
                score_max=self.score_max,
                reasoning_hidden=reasoning_hidden,
                y_raw=y_raw,
            )
        attn = aux.get("attention_weights")
        if isinstance(attn, torch.Tensor) and attn.numel():
            attn_cpu = attn.detach().float().cpu()
            self._last_neural_attention_stats = {
                "mean": float(attn_cpu.mean().item()),
                "max": float(attn_cpu.max().item()),
                "min": float(attn_cpu.min().item()),
            }
        else:
            self._last_neural_attention_stats = {"mean": 0.0, "max": 0.0, "min": 0.0}
        z_flat = z[0].detach().cpu().float()
        if not torch.isfinite(z_flat).all():
            z_flat = torch.nan_to_num(z_flat, nan=0.0, posinf=10.0, neginf=-10.0)
        expected_dim = self.evidence_dim(anchor_hiddens.shape[0])
        if z_flat.numel() != expected_dim:
            raise ValueError(f"Neural evidence dim mismatch: got {z_flat.numel()}, expected {expected_dim}")
        return z_flat

    def _validate_evidence_vector(self, z: torch.Tensor, n_anchors: int) -> torch.Tensor:
        expected_dim = self.evidence_dim(n_anchors)
        if z.numel() != expected_dim:
            raise ValueError(f"Evidence dim mismatch: got {z.numel()}, expected {expected_dim}")
        if not torch.isfinite(z).all():
            z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        enabled = self.enabled_evidence_blocks()
        if not all(bool(v) for v in enabled.values()):
            z = z.clone()
            for block, indices in self.evidence_block_slices(n_anchors).items():
                if not bool(enabled.get(block, True)) and indices:
                    z[indices] = 0.0
        return z

    def compute_anchor_geometry(
        self,
        *,
        anchor_hiddens: torch.Tensor,
        essay_hiddens: torch.Tensor,
        y_true: List[int],
        anchor_scores: List[int],
    ) -> Dict[str, float]:
        """Compute Phase-2 anchor geometry diagnostics.

        Scores are normalized into a compact [0, 1]-ish range so gamma can be
        used directly as a small additive fitness weight.
        """
        anchors = F.normalize(anchor_hiddens.float(), p=2, dim=1)
        essays = F.normalize(essay_hiddens.float(), p=2, dim=1)
        if anchors.shape[0] < 3 or essays.shape[0] == 0:
            return {
                "anchor_geometry_score": 0.0,
                "anchor_separation": 0.0,
                "anchor_separation_raw": 0.0,
                "anchor_ordinal_consistency": 0.0,
                "anchor_monotonicity": 0.0,
                "anchor_min_pair": [],
            }

        pairwise = 1.0 - torch.mm(anchors, anchors.T)
        tri = torch.triu_indices(pairwise.shape[0], pairwise.shape[1], offset=1)
        pair_vals = pairwise[tri[0], tri[1]]
        min_pos = int(pair_vals.argmin().detach().cpu())
        sep_raw = float(pair_vals[min_pos].detach().cpu())
        min_pair = [int(tri[0][min_pos].item()), int(tri[1][min_pos].item())]
        anchor_separation = max(0.0, min(1.0, sep_raw / 2.0))

        distances = 1.0 - torch.mm(essays, anchors.T)
        closest = distances.argmin(dim=1).cpu().tolist()
        true_bands = [self._score_band_index(y, anchor_scores=anchor_scores) for y in y_true]
        ordinal = sum(int(p == t) for p, t in zip(closest, true_bands)) / max(1, len(true_bands))

        sims = torch.mm(essays, anchors.T)
        mono_values = (sims[:, -1] - sims[:, 0]).detach().cpu().numpy()
        corr = self._spearman_corr(mono_values, np.asarray(y_true, dtype=np.float32))
        monotonicity = (corr + 1.0) / 2.0

        score = 0.3 * anchor_separation + 0.4 * ordinal + 0.3 * monotonicity
        return {
            "anchor_geometry_score": float(score),
            "anchor_separation": float(anchor_separation),
            "anchor_separation_raw": float(sep_raw),
            "anchor_ordinal_consistency": float(ordinal),
            "anchor_monotonicity": float(monotonicity),
            "anchor_min_pair": min_pair,
        }

    def _early_rejection_check(
        self,
        *,
        items: List[Dict],
        results: List[ScoringResult],
        anchor_hiddens: torch.Tensor,
    ) -> Dict:
        y_true = [item["domain1_score"] for item in items]
        y_raw = [result.y_raw for result in results]
        early_qwk = self._safe_qwk(y_true, y_raw)
        collapse_ratio = self._score_collapse_ratio(y_raw)
        anchor_sep_raw = self._anchor_separation_raw(anchor_hiddens)
        reasons = []
        if collapse_ratio >= self.config.score_collapse_threshold:
            reasons.append(f"score_collapse={collapse_ratio:.3f}")
        if anchor_sep_raw < self.config.anchor_sep_threshold:
            reasons.append(f"anchor_sep={anchor_sep_raw:.4f}")
        if early_qwk < self.config.early_qwk_threshold:
            reasons.append(f"early_qwk={early_qwk:.4f}")
        metrics = {
            "early_qwk": early_qwk,
            "score_collapse_ratio": collapse_ratio,
            "anchor_separation_raw": anchor_sep_raw,
            "n_mini": len(results),
        }
        return {
            "rejected": bool(reasons),
            "reason": "; ".join(reasons),
            "metrics": metrics,
        }

    def classify_error_types(
        self,
        *,
        items: List[Dict],
        results: List[ScoringResult],
        y_true: List[int],
        anchor_hiddens: torch.Tensor,
        anchor_scores: List[int],
        anchor_geometry: Dict[str, float],
    ) -> Dict:
        anchors = F.normalize(anchor_hiddens.float(), p=2, dim=1)
        diagnostics = []
        summary: Dict[str, int] = {}
        y_raw = [r.y_raw for r in results]
        collapse_ratio = self._score_collapse_ratio(y_raw)
        if collapse_ratio >= self.config.score_collapse_threshold:
            summary[ErrorType.RAW_COLLAPSE.value] = 1

        for item, result, truth in zip(items, results, y_true):
            if result.hidden is None:
                continue
            h = F.normalize(result.hidden.float(), p=2, dim=0)
            sims = torch.mv(anchors, h).detach().cpu().numpy()
            closest = int(np.argmax(sims))
            true_band = self._score_band_index(truth, anchor_scores=anchor_scores)
            error = int(result.y_raw) - int(truth)
            err_type = self._classify_single_error(error, closest, result.raw_text)
            if err_type is None and closest != true_band:
                err_type = ErrorType.BOUNDARY_AMBIGUITY if abs(error) <= 1 else ErrorType.GENERAL_ERROR
            if err_type is None:
                continue
            summary[err_type.value] = summary.get(err_type.value, 0) + 1
            diagnostics.append(
                {
                    "essay_id": item.get("essay_id"),
                    "y_true": int(truth),
                    "y_raw": int(result.y_raw),
                    "error": int(error),
                    "closest_anchor": closest,
                    "true_band": true_band,
                    "error_type": err_type.value,
                    "anchor_sims": [float(x) for x in sims.tolist()],
                    "reasoning_preview": result.raw_text[:240],
                }
            )

        diagnostics.sort(key=lambda x: (abs(x["error"]), x["error_type"]), reverse=True)
        diagnostics = diagnostics[: self.config.evidence_diagnostics_top_n]

        if anchor_geometry.get("anchor_separation_raw", 1.0) < self.config.anchor_sep_threshold:
            summary[ErrorType.ANCHOR_CONFUSION.value] = summary.get(ErrorType.ANCHOR_CONFUSION.value, 0) + 1

        dominant = max(summary, key=summary.get) if summary else ""
        suggested_slot = self._suggest_anchor_slot(dominant, anchor_geometry, summary)
        return {
            "diagnostics": diagnostics,
            "summary": summary,
            "dominant_error_type": dominant,
            "suggested_anchor_mutation_slot": suggested_slot,
        }

    def _classify_single_error(self, error: int, closest_anchor: int, raw_text: str) -> Optional[ErrorType]:
        lowered = raw_text.lower()
        weak_terms = ("weak", "limited", "unclear", "insufficient", "confusing", "underdeveloped")
        strong_terms = ("strong", "clear", "well-developed", "coherent", "effective")
        if error >= 2 and closest_anchor == 0:
            return ErrorType.OVER_SCORE_LOW_HIDDEN
        if error <= -2 and closest_anchor >= 2:
            return ErrorType.UNDER_SCORE_HIGH_HIDDEN
        if error >= 2 and any(term in lowered for term in weak_terms):
            return ErrorType.REASONING_CONTRADICTION
        if error <= -2 and any(term in lowered for term in strong_terms):
            return ErrorType.REASONING_CONTRADICTION
        if abs(error) == 1:
            return ErrorType.BOUNDARY_AMBIGUITY
        if error != 0:
            return ErrorType.GENERAL_ERROR
        return None

    def _suggest_anchor_slot(
        self,
        dominant_error_type: str,
        anchor_geometry: Dict,
        summary: Optional[Dict[str, int]] = None,
    ) -> Optional[int]:
        if dominant_error_type == ErrorType.OVER_SCORE_LOW_HIDDEN.value:
            return 0
        if dominant_error_type == ErrorType.UNDER_SCORE_HIGH_HIDDEN.value:
            return 2
        if dominant_error_type == ErrorType.BOUNDARY_AMBIGUITY.value:
            return 1
        if dominant_error_type == ErrorType.ANCHOR_CONFUSION.value:
            pair = anchor_geometry.get("anchor_min_pair") or []
            if len(pair) == 2:
                return int(pair[1])
        summary = summary or {}
        if summary.get(ErrorType.UNDER_SCORE_HIGH_HIDDEN.value, 0) >= 3:
            return 2
        if summary.get(ErrorType.OVER_SCORE_LOW_HIDDEN.value, 0) >= 3:
            return 0
        if summary.get(ErrorType.BOUNDARY_AMBIGUITY.value, 0) >= 3:
            return 1
        return None

    def _anchor_separation_raw(self, anchor_hiddens: torch.Tensor) -> float:
        anchors = F.normalize(anchor_hiddens.float(), p=2, dim=1)
        if anchors.shape[0] < 2:
            return 0.0
        pairwise = 1.0 - torch.mm(anchors, anchors.T)
        tri = torch.triu_indices(pairwise.shape[0], pairwise.shape[1], offset=1)
        return float(pairwise[tri[0], tri[1]].min().detach().cpu())

    def _score_collapse_ratio(self, scores: List[int]) -> float:
        if not scores:
            return 0.0
        counts: Dict[int, int] = {}
        for score in scores:
            counts[int(score)] = counts.get(int(score), 0) + 1
        return max(counts.values()) / len(scores)

    def _cost_scale(self, total_sec: float) -> float:
        if total_sec <= 0:
            return 0.0
        return math.log1p(total_sec) / 10.0

    def _anchor_role_names(self, n_anchors: int) -> List[str]:
        base = ["low", "mid", "high"] if n_anchors <= 3 else ["low", "lower_boundary", "upper_boundary", "high"]
        if n_anchors <= len(base):
            return base[:n_anchors]
        return base + [f"extra_{i}" for i in range(len(base), n_anchors)]

    def _select_mini_items(self, items: List[Dict], n_mini: int) -> List[Dict]:
        """Pick an early-rejection mini-set that covers low/mid/high score bands."""
        if n_mini <= 0 or n_mini >= len(items):
            return list(items)
        buckets: Dict[int, List[Dict]] = {0: [], 1: [], 2: []}
        for item in items:
            buckets[self._score_band_index(item["domain1_score"])].append(item)

        selected = []
        seen = set()
        while len(selected) < n_mini and any(buckets.values()):
            progressed = False
            for band in (0, 1, 2):
                if len(selected) >= n_mini:
                    break
                if not buckets[band]:
                    continue
                item = buckets[band].pop(0)
                essay_id = item.get("essay_id")
                if essay_id in seen:
                    continue
                selected.append(item)
                seen.add(essay_id)
                progressed = True
            if not progressed:
                break
        if len(selected) < n_mini:
            for item in items:
                essay_id = item.get("essay_id")
                if essay_id in seen:
                    continue
                selected.append(item)
                seen.add(essay_id)
                if len(selected) >= n_mini:
                    break
        return selected

    def _score_band_label(self, score: int) -> str:
        lo = self.score_min + (self.score_max - self.score_min) * 0.33
        hi = self.score_min + (self.score_max - self.score_min) * 0.66
        if score <= lo:
            return "low"
        if score >= hi:
            return "high"
        return "mid"

    def _score_band_index(self, score: int, anchor_scores: Optional[List[int]] = None) -> int:
        if anchor_scores and len(anchor_scores) >= 3:
            low_s, mid_s, high_s = [float(x) for x in anchor_scores[:3]]
            lo_cut = (low_s + mid_s) / 2.0
            hi_cut = (mid_s + high_s) / 2.0
            if score <= lo_cut:
                return 0
            if score >= hi_cut:
                return 2
            return 1
        label = self._score_band_label(score)
        if label == "low":
            return 0
        if label == "high":
            return 2
        return 1

    def _spearman_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2:
            return 0.0
        rx = self._rankdata(x.astype(np.float64))
        ry = self._rankdata(y.astype(np.float64))
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
        if denom <= 0.0:
            return 0.0
        corr = float(np.sum(rx * ry) / denom)
        return corr if np.isfinite(corr) else 0.0

    def _rankdata(self, values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and sorted_values[j] == sorted_values[i]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            ranks[order[i:j]] = avg_rank
            i = j
        return ranks

    def _safe_qwk(self, y_true: List[int], y_pred: List[int]) -> float:
        try:
            qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        except Exception:
            return 0.0
        return qwk if np.isfinite(qwk) else 0.0

    def _prediction_distribution_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
    ) -> Dict:
        scores = list(range(self.score_min, self.score_max + 1))
        n = max(1, len(y_true))
        true_counts = {str(s): 0 for s in scores}
        pred_counts = {str(s): 0 for s in scores}
        for y in y_true:
            key = str(int(y))
            if key in true_counts:
                true_counts[key] += 1
        for y in y_pred:
            key = str(int(y))
            if key in pred_counts:
                pred_counts[key] += 1

        true_probs = np.array([true_counts[str(s)] / n for s in scores], dtype=float)
        pred_probs = np.array([pred_counts[str(s)] / n for s in scores], dtype=float)
        tv_distance = float(0.5 * np.abs(pred_probs - true_probs).sum())
        pred_collapse_ratio = float(pred_probs.max()) if len(pred_probs) else 0.0
        true_max_score_rate = float(true_counts[str(self.score_max)] / n)
        pred_max_score_rate = float(pred_counts[str(self.score_max)] / n)
        smoothed_true_max = (true_counts[str(self.score_max)] + 1.0) / (n + len(scores))
        smoothed_pred_max = (pred_counts[str(self.score_max)] + 1.0) / (n + len(scores))
        max_score_overprediction_ratio = float(smoothed_pred_max / max(1e-8, smoothed_true_max))
        true_mean = float(np.mean(y_true)) if y_true else 0.0
        pred_mean = float(np.mean(y_pred)) if y_pred else 0.0
        pred_span = int(max(y_pred) - min(y_pred)) if y_pred else 0
        return {
            "true_counts": true_counts,
            "pred_counts": pred_counts,
            "tv_distance": tv_distance,
            "pred_collapse_ratio": pred_collapse_ratio,
            "true_max_score_rate": true_max_score_rate,
            "pred_max_score_rate": pred_max_score_rate,
            "max_score_overprediction_ratio": max_score_overprediction_ratio,
            "true_mean": true_mean,
            "pred_mean": pred_mean,
            "pred_bias": pred_mean - true_mean,
            "pred_span": pred_span,
        }

    def _distribution_penalty(self, metrics: Dict) -> float:
        if not self.config.distribution_guard_enabled:
            return 0.0
        tv_excess = max(0.0, float(metrics.get("tv_distance", 0.0)) - self.config.distribution_tv_threshold)
        collapse_excess = max(
            0.0,
            float(metrics.get("pred_collapse_ratio", 0.0)) - self.config.score_collapse_threshold,
        )
        max_score_excess = max(
            0.0,
            float(metrics.get("max_score_overprediction_ratio", 0.0))
            - self.config.max_score_overprediction_threshold,
        )
        return (
            self.config.distribution_penalty_weight * tv_excess
            + self.config.collapse_penalty_weight * collapse_excess
            + self.config.max_score_penalty_weight * max_score_excess
        )

    def _distribution_flat_payload(self, metrics: Dict) -> Dict:
        return {
            "pace_distribution_tv": metrics.get("tv_distance", 0.0),
            "pace_pred_collapse_ratio": metrics.get("pred_collapse_ratio", 0.0),
            "pace_true_max_score_rate": metrics.get("true_max_score_rate", 0.0),
            "pace_pred_max_score_rate": metrics.get("pred_max_score_rate", 0.0),
            "pace_max_score_overprediction_ratio": metrics.get("max_score_overprediction_ratio", 0.0),
            "pace_pred_bias": metrics.get("pred_bias", 0.0),
            "pace_pred_span": metrics.get("pred_span", 0),
        }

    def _train_calibrator(
        self,
        z_calib: torch.Tensor,
        y_calib: torch.Tensor,
    ) -> CoralOrdinalCalibrator:
        """在 calib 集上训练轻量 CoralOrdinalCalibrator（无 dropout，全批量）。"""
        cfg = CalibratorConfig(
            input_dim=z_calib.shape[1],
            score_min=self.score_min,
            score_max=self.score_max,
            hidden_dim=self.config.hidden_dim,
            dropout=0.0,           # calib 集极小，禁止 dropout 避免欠拟合
            lambda_qwk=self.config.lambda_qwk,
            lambda_reg=1e-4,
        )
        model = CoralOrdinalCalibrator(cfg).to(self._device)
        z_dev = z_calib.to(self._device)
        y_dev = y_calib.to(self._device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)

        model.train()
        for _ in range(self.config.epochs):
            loss, _ = calibrator_loss(model, z_dev, y_dev)
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    def _usage_payload(self, usage_before: Dict[str, int]) -> Dict:
        usage_delta = self.backend.usage_delta(usage_before)
        return {
            "local_usage": usage_delta,
            "local_prompt_tokens": usage_delta.get("prompt_tokens", 0),
            "local_completion_tokens": usage_delta.get("completion_tokens", 0),
            "local_representation_tokens": usage_delta.get("representation_tokens", 0),
        }

    def _fallback_result(
        self,
        raw_qwk: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> Dict:
        """PACE cannot run; caller uses full-val raw QWK for selection."""
        evidence_payload = self.evidence_metadata()
        result = {
            "pace_qwk": 0.0,
            "raw_qwk": raw_qwk,
            "anchor_geometry_score": 0.0,
            "anchor_separation": 0.0,
            "anchor_separation_raw": 0.0,
            "anchor_ordinal_consistency": 0.0,
            "anchor_monotonicity": 0.0,
            "anchor_min_pair": [],
            "pace_diagnostics": [],
            "pace_diagnostic_summary": {},
            "dominant_error_type": "",
            "suggested_anchor_mutation_slot": None,
            "early_rejection_metrics": {},
            "cost_penalty": 0.0,
            "evidence_dim": self.evidence_dim(),
            "evidence_schema": self.evidence_schema(),
            **evidence_payload,
            "calibrator_probe_combined": 0.0,
            "pace_qwk_used_for_selection": False,
            "anchor_inference_sec": 0.0,
            "calib_inference_sec": 0.0,
            "fitness_inference_sec": 0.0,
            "calibrator_train_sec": 0.0,
            "total_pace_sec": 0.0,
            "local_usage": {},
            "local_prompt_tokens": 0,
            "local_completion_tokens": 0,
            "local_representation_tokens": 0,
            "n_calib": 0,
            "n_fitness": 0,
            "scoring_forward_passes": 0,
            "representation_forward_passes": 0,
            "total_local_forward_passes": 0,
            "_fallback": True,
        }
        if extra:
            result.update(extra)
        return result
