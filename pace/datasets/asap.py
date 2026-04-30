"""ASAP data loader aligned with wise_aes.py / eval_5fold.py fold logic.

IMPORTANT: split semantics must match eval_5fold.py (KFold, random_state=42)
so that per-fold reconstruction of train / val / test matches the original
WISE-AES experiment exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import KFold

DEFAULT_TSV = "data/raw/training_set_rel3.tsv"
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5


def load_asap(
    essay_set: int,
    tsv_path: str = DEFAULT_TSV,
    repo_root: Path | str | None = None,
) -> List[Dict]:
    """Load a single ASAP prompt as the list-of-dict format wise_aes.py expects.

    The returned order is the DataFrame iteration order (preserved) so that
    `KFold(n_splits=5, shuffle=True, random_state=42).split(all_data)` produces
    exactly the same indices as wise_aes.py main + eval_5fold.py.
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    path = root / tsv_path if not Path(tsv_path).is_absolute() else Path(tsv_path)
    df = pd.read_csv(path, sep="\t", encoding="latin-1")
    df = df[df["essay_set"] == essay_set]
    records: List[Dict] = []
    for _, row in df.iterrows():
        records.append(
            {
                "essay_id": row["essay_id"],
                "essay_text": row["essay"],
                "domain1_score": int(row["domain1_score"]),
                "meta": {"essay_set": int(row["essay_set"])},
            }
        )
    return records


def get_fold_splits(
    n: int,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> List[Tuple[list, list]]:
    """Return the list of (train_val_idx, test_idx) tuples used across the project."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # sklearn's KFold only needs a range array
    return list(kf.split(list(range(n))))


def split_for_fold(all_data: List[Dict], fold_idx: int) -> Tuple[List[Dict], List[Dict]]:
    """Return (train_val_set, test_set) for the given fold, matching eval_5fold.py."""
    folds = get_fold_splits(len(all_data))
    train_val_idx, test_idx = folds[fold_idx]
    train_val = [all_data[i] for i in train_val_idx]
    test = [all_data[i] for i in test_idx]
    return train_val, test


def split_train_val_test_for_fold(
    all_data: List[Dict],
    fold_idx: int,
    train_ratio_within_train_val: float = 0.8,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Return (train_set, val_set, test_set) using wise_aes.py split semantics.

    WISE-AES first applies 5-fold KFold to get ``train_val`` vs ``test``, then
    splits the *ordered* ``train_val_idx`` list into 80% train and 20% val
    without shuffling. This helper mirrors lines 977-987 in ``wise_aes.py`` so
    that any PACE calibration split remains directly comparable.
    """
    folds = get_fold_splits(len(all_data))
    train_val_idx, test_idx = folds[fold_idx]
    split = int(len(train_val_idx) * train_ratio_within_train_val)
    train_idx = train_val_idx[:split]
    val_idx = train_val_idx[split:]
    train = [all_data[i] for i in train_idx]
    val = [all_data[i] for i in val_idx]
    test = [all_data[i] for i in test_idx]
    return train, val, test
