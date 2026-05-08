"""Base dataset adapter contracts for AES datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import KFold, train_test_split


@dataclass(frozen=True)
class AESExample:
    essay_id: int | str
    prompt_id: int | str
    essay_text: str
    score: int | float
    score_min: int | float
    score_max: int | float
    rubric: str = ""
    prompt_text: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class AESDatasetAdapter(ABC):
    """Common adapter interface for ASAP, TOEFL11, EssayJudge, and others."""

    @abstractmethod
    def load(self) -> Sequence[AESExample]:
        raise NotImplementedError

    @abstractmethod
    def get_prompt_ids(self) -> List[int | str]:
        raise NotImplementedError

    @abstractmethod
    def get_score_range(self, prompt_id: int | str) -> Tuple[int | float, int | float]:
        raise NotImplementedError

    @abstractmethod
    def get_rubric(self, prompt_id: int | str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_prompt_text(self, prompt_id: int | str) -> str:
        raise NotImplementedError

    @abstractmethod
    def split_train_val_test(
        self,
        prompt_id: int | str,
        fold: int,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        raise NotImplementedError


class ASAPCompatAdapter(AESDatasetAdapter):
    """Minimal ASAP wrapper that preserves the current wise_aes.py row shape."""

    def __init__(self, data_path: str | Path, *, seed: int = 42, n_splits: int = 5) -> None:
        self.data_path = Path(data_path)
        self.seed = int(seed)
        self.n_splits = int(n_splits)
        self._df: pd.DataFrame | None = None

    def _load_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.data_path, sep="\t", encoding="latin1")
        return self._df

    def load(self) -> Sequence[AESExample]:
        df = self._load_df()
        examples: List[AESExample] = []
        for prompt_id in self.get_prompt_ids():
            score_min, score_max = self.get_score_range(prompt_id)
            sub = df[df["essay_set"] == int(prompt_id)]
            for row in sub.itertuples(index=False):
                examples.append(
                    AESExample(
                        essay_id=getattr(row, "essay_id"),
                        prompt_id=int(prompt_id),
                        essay_text=str(getattr(row, "essay")),
                        score=int(getattr(row, "domain1_score")),
                        score_min=score_min,
                        score_max=score_max,
                    )
                )
        return examples

    def get_prompt_ids(self) -> List[int | str]:
        df = self._load_df()
        return sorted(int(x) for x in df["essay_set"].dropna().unique())

    def get_score_range(self, prompt_id: int | str) -> Tuple[int | float, int | float]:
        df = self._load_df()
        sub = df[df["essay_set"] == int(prompt_id)]
        if sub.empty:
            raise ValueError(f"prompt_id {prompt_id!r} not found")
        return int(sub["domain1_score"].min()), int(sub["domain1_score"].max())

    def get_rubric(self, prompt_id: int | str) -> str:
        return ""

    def get_prompt_text(self, prompt_id: int | str) -> str:
        return ""

    def split_train_val_test(
        self,
        prompt_id: int | str,
        fold: int,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        df = self._load_df()
        sub = df[df["essay_set"] == int(prompt_id)].copy()
        if sub.empty:
            raise ValueError(f"prompt_id {prompt_id!r} not found")
        rows = [
            {
                "essay_id": int(row.essay_id),
                "essay_set": int(row.essay_set),
                "essay_text": str(row.essay),
                "domain1_score": int(row.domain1_score),
            }
            for row in sub.itertuples(index=False)
        ]
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        splits = list(kf.split(rows))
        train_val_idx, test_idx = splits[int(fold) % len(splits)]
        train_val = [rows[i] for i in train_val_idx]
        test = [rows[i] for i in test_idx]
        train, val = train_test_split(
            train_val,
            test_size=0.2,
            random_state=self.seed + int(fold),
        )
        return list(train), list(val), list(test)
