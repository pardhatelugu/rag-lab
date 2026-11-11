"""STEP 1: similarity. Fill in the two functions, then run: pytest tests/test_step1.py"""
from __future__ import annotations

import numpy as np


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return an (len(a), len(b)) matrix of cosine similarities.

    Inputs are NOT guaranteed to be normalized. Guard against zero vectors.
    Hint: normalize each row, then one matrix multiply.
    """
    raise NotImplementedError


def top_k(sims: np.ndarray, k: int) -> np.ndarray:
    """For each row of `sims`, return the indices of its k largest values, best first.

    If k > number of columns, clip it. Result shape: (n_rows, min(k, n_cols)).
    Hint: np.argpartition is O(n) vs a full argsort O(n log n). Why might that matter at 10M vectors?
    """
    raise NotImplementedError
