from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis.profile_asap_prompt import (
    profile_prompt_dataframe,
    profile_type_from_span,
)


def test_profile_type_rules():
    assert profile_type_from_span(6) == "narrow"
    assert profile_type_from_span(7) == "medium"
    assert profile_type_from_span(20) == "wide"
    assert profile_type_from_span(40) == "ultrawide"


def test_prompt_profiler_toy_dataframe():
    df = pd.DataFrame(
        {
            "essay_id": list(range(8)),
            "essay_set": [1] * 8,
            "essay": [
                "short essay",
                "another short essay",
                "middle essay text",
                "more middle essay text",
                "high quality essay text",
                "top essay text",
                "weak text",
                "excellent organized fluent essay",
            ],
            "domain1_score": [2, 3, 4, 5, 6, 7, 8, 8],
        }
    )
    profile = profile_prompt_dataframe(df, prompt=1)
    assert profile["score_min"] == 2
    assert profile["score_max"] == 8
    assert profile["profile_type"] == "narrow"
    assert profile["max_score_count"] == 2
    assert profile["low_anchor_pool_count"] > 0
    assert profile["high_anchor_pool_count"] > 0
