"""Protocol-level data structures for WISE-PACE / HE-guided PE.

These classes keep the paper-facing objects explicit without forcing a large
rewrite of the current optimizer. Runtime code may still use PromptIndividual,
but every candidate can now be serialized as a protocol candidate with lineage,
diagnostic, mutation, and anchor-bank metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class DiagnosticType(str, Enum):
    HIGH_TAIL_UNDERSCORE = "HIGH_TAIL_UNDERSCORE"
    LOW_TAIL_OVERSCORE = "LOW_TAIL_OVERSCORE"
    SCORE_COMPRESSION = "SCORE_COMPRESSION"
    BOUNDARY_AMBIGUITY = "BOUNDARY_AMBIGUITY"
    ANCHOR_CONFUSION = "ANCHOR_CONFUSION"
    RUBRIC_UNDER_SPECIFICATION = "RUBRIC_UNDER_SPECIFICATION"
    FORMAT_INSTABILITY = "FORMAT_INSTABILITY"
    NO_CLEAR_SIGNAL = "NO_CLEAR_SIGNAL"


class MutationOperator(str, Enum):
    MAX_SCORE_CONTRASTIVE = "max_score_contrastive_mutation"
    HIGH_TAIL_INSTRUCTION = "high_tail_instruction_mutation"
    BOUNDARY_CLARIFICATION = "boundary_clarification_mutation"
    SCORE_MAPPING = "score_mapping_mutation"
    ANCHOR_SLOT = "anchor_slot_mutation"
    NEGATIVE_CONSTRAINT = "negative_constraint_mutation"
    SCORE_DISTRIBUTION = "score_distribution_mutation"
    GENERAL_REFLECTION = "general_reflection_mutation"


@dataclass
class EssayAnchor:
    essay_id: int | str
    score: int
    essay_text: str = ""
    role: str = "absolute"

    @classmethod
    def from_example(cls, example: Dict[str, Any], role: str = "absolute") -> "EssayAnchor":
        return cls(
            essay_id=example.get("essay_id"),
            score=int(example.get("domain1_score", example.get("score", 0))),
            essay_text=str(example.get("essay_text", "")),
            role=role,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContrastivePair:
    boundary: str
    lower_anchor: EssayAnchor
    upper_anchor: EssayAnchor
    rationale_diff: str = ""

    @property
    def lower_score(self) -> int:
        return int(self.lower_anchor.score)

    @property
    def upper_score(self) -> int:
        return int(self.upper_anchor.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary": self.boundary,
            "lower_anchor": self.lower_anchor.to_dict(),
            "upper_anchor": self.upper_anchor.to_dict(),
            "lower_score": self.lower_score,
            "upper_score": self.upper_score,
            "rationale_diff": self.rationale_diff,
        }


@dataclass
class AnchorBank:
    absolute_anchors: List[EssayAnchor] = field(default_factory=list)
    contrastive_pairs: List[ContrastivePair] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "absolute_anchors": [a.to_dict() for a in self.absolute_anchors],
            "contrastive_pairs": [p.to_dict() for p in self.contrastive_pairs],
        }


@dataclass
class ProtocolCandidate:
    id: str
    parent_id: Optional[str]
    instruction: str
    anchor_bank: AnchorBank
    generation_method: str = ""
    diagnostic_source: str = ""
    diagnostic_type: str = DiagnosticType.NO_CLEAR_SIGNAL.value
    mutation_operator: str = MutationOperator.GENERAL_REFLECTION.value
    cost: Dict[str, Any] = field(default_factory=dict)
    val_diag_metrics: Dict[str, Any] = field(default_factory=dict)
    val_sel_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    protocol_diff: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "instruction": self.instruction,
            "anchor_bank": self.anchor_bank.to_dict(),
            "generation_method": self.generation_method,
            "diagnostic_source": self.diagnostic_source,
            "diagnostic_type": self.diagnostic_type,
            "mutation_operator": self.mutation_operator,
            "cost": self.cost,
            "val_diag_metrics": self.val_diag_metrics,
            "val_sel_metrics": self.val_sel_metrics,
            "test_metrics": self.test_metrics,
            "evidence_summary": self.evidence_summary,
            "protocol_diff": self.protocol_diff,
        }


def canonical_diagnostic_type(raw_type: str | None, raw_metrics: Optional[Dict[str, Any]] = None) -> DiagnosticType:
    """Map heterogeneous diagnostics to the paper-facing failure taxonomy."""
    raw = str(raw_type or "").strip().lower()
    metrics = raw_metrics or {}
    if raw in {"under_score_high_hidden", "high_tail_under_score", "high_tail_underscore"}:
        return DiagnosticType.HIGH_TAIL_UNDERSCORE
    if raw in {"over_score_low_hidden", "low_tail_overscore"}:
        return DiagnosticType.LOW_TAIL_OVERSCORE
    if raw in {"raw_collapse", "score_compression", "score_distribution_collapse"}:
        return DiagnosticType.SCORE_COMPRESSION
    if raw == "boundary_ambiguity":
        return DiagnosticType.BOUNDARY_AMBIGUITY
    if raw == "anchor_confusion":
        return DiagnosticType.ANCHOR_CONFUSION
    if raw == "format_instability":
        return DiagnosticType.FORMAT_INSTABILITY
    if raw == "reasoning_score_contradiction":
        return DiagnosticType.RUBRIC_UNDER_SPECIFICATION

    high_recall = float(metrics.get("high_score_recall", metrics.get("high_recall", 1.0)) or 0.0)
    max_recall = float(metrics.get("max_score_recall", 1.0) or 0.0)
    n_true_max = int(metrics.get("n_true_max_score", metrics.get("max_score_true_count", 0)) or 0)
    collapse = float(metrics.get("pred_collapse_ratio", 0.0) or 0.0)
    pred_span = int(metrics.get("pred_span", 999) or 999)
    if n_true_max > 0 and max_recall < 1.0:
        return DiagnosticType.HIGH_TAIL_UNDERSCORE
    if high_recall < 0.30:
        return DiagnosticType.HIGH_TAIL_UNDERSCORE
    if collapse >= 0.70 or pred_span <= 2:
        return DiagnosticType.SCORE_COMPRESSION
    return DiagnosticType.NO_CLEAR_SIGNAL


def mutation_operator_for_diagnostic(diagnostic_type: DiagnosticType) -> MutationOperator:
    mapping = {
        DiagnosticType.HIGH_TAIL_UNDERSCORE: MutationOperator.HIGH_TAIL_INSTRUCTION,
        DiagnosticType.LOW_TAIL_OVERSCORE: MutationOperator.NEGATIVE_CONSTRAINT,
        DiagnosticType.SCORE_COMPRESSION: MutationOperator.SCORE_DISTRIBUTION,
        DiagnosticType.BOUNDARY_AMBIGUITY: MutationOperator.BOUNDARY_CLARIFICATION,
        DiagnosticType.ANCHOR_CONFUSION: MutationOperator.ANCHOR_SLOT,
        DiagnosticType.RUBRIC_UNDER_SPECIFICATION: MutationOperator.SCORE_MAPPING,
        DiagnosticType.FORMAT_INSTABILITY: MutationOperator.GENERAL_REFLECTION,
        DiagnosticType.NO_CLEAR_SIGNAL: MutationOperator.GENERAL_REFLECTION,
    }
    return mapping.get(diagnostic_type, MutationOperator.GENERAL_REFLECTION)


def contrastive_pair_from_dict(pair: Dict[str, Any]) -> ContrastivePair:
    lower = pair.get("lower_anchor", {})
    upper = pair.get("upper_anchor", {})
    return ContrastivePair(
        boundary=str(pair.get("boundary") or f"{lower.get('score')}_vs_{upper.get('score')}"),
        lower_anchor=EssayAnchor(
            essay_id=lower.get("essay_id"),
            score=int(lower.get("score", lower.get("domain1_score", 0))),
            essay_text=str(lower.get("essay_text", "")),
            role=str(lower.get("role", "contrastive_lower")),
        ),
        upper_anchor=EssayAnchor(
            essay_id=upper.get("essay_id"),
            score=int(upper.get("score", upper.get("domain1_score", 0))),
            essay_text=str(upper.get("essay_text", "")),
            role=str(upper.get("role", "contrastive_upper")),
        ),
        rationale_diff=str(pair.get("rationale_diff", "")),
    )


def protocol_diff_summary(
    parent: Optional[Dict[str, Any]],
    child: Dict[str, Any],
) -> Dict[str, Any]:
    if not parent:
        return {"instruction_changed": False, "anchors_changed": False, "contrastive_anchors_changed": False}
    parent_abs = parent.get("static_exemplar_ids") or parent.get("absolute_anchor_ids") or []
    child_abs = child.get("static_exemplar_ids") or child.get("absolute_anchor_ids") or []
    parent_ctr = parent.get("contrastive_anchor_pair_ids") or []
    child_ctr = child.get("contrastive_anchor_pair_ids") or []
    return {
        "instruction_changed": parent.get("full_instruction") != child.get("full_instruction"),
        "anchors_changed": list(parent_abs) != list(child_abs),
        "contrastive_anchors_changed": list(parent_ctr) != list(child_ctr),
        "changed_absolute_anchor_ids_before": list(parent_abs),
        "changed_absolute_anchor_ids_after": list(child_abs),
        "changed_contrastive_pair_ids_before": list(parent_ctr),
        "changed_contrastive_pair_ids_after": list(child_ctr),
    }
