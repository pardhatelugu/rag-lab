from __future__ import annotations

import numpy as np


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(n == 0, 1.0, n)

    return norm(a) @ norm(b).T


def top_k(sims: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sims.shape[1])
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)
