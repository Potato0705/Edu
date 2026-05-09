"""PACE-AES: Protocol-conditioned Anchor-relative Calibration Evolution for AES.

Layer 2 companion to WISE-AES (see D:/SZTU/Education for AI/PACE_AES_method_skeleton.md).
All PACE code lives under this package; the existing wise_aes.py scoring path is
imported unchanged.
"""

__version__ = "0.1.0.dev0"

from .protocol import (
    AnchorBank,
    ContrastivePair,
    DiagnosticType,
    EssayAnchor,
    MutationOperator,
    ProtocolCandidate,
    canonical_diagnostic_type,
    mutation_operator_for_diagnostic,
    protocol_diff_summary,
)

__all__ = [
    "AnchorBank",
    "ContrastivePair",
    "DiagnosticType",
    "EssayAnchor",
    "MutationOperator",
    "ProtocolCandidate",
    "canonical_diagnostic_type",
    "mutation_operator_for_diagnostic",
    "protocol_diff_summary",
]
