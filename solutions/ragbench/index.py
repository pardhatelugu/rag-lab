"""Vector index (exact or HNSW), BM25, and reciprocal-rank fusion."""
from __future__ import annotations

import re

import numpy as np


class VectorIndex:
    """backend='flat' is exact cosine search (numpy). backend='hnsw' uses FAISS
    HNSW (approximate) so you can measure the recall/latency trade-off."""

    def __init__(self, backend: str = "flat", m: int = 32, ef_search: int = 64):
        self.backend, self.m, self.ef = backend, m, ef_search
        self.vecs = None
        self.faiss = None

    def add(self, vecs: np.ndarray) -> None:
        if self.backend == "flat":
            self.vecs = vecs
        elif self.backend == "hnsw":
            import faiss

            self.faiss = faiss.IndexHNSWFlat(vecs.shape[1], self.m, faiss.METRIC_INNER_PRODUCT)
            self.faiss.hnsw.efSearch = self.ef
            self.faiss.add(vecs)
        else:
            raise ValueError(self.backend)

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        """Return (n_queries, k) array of chunk indices, best first."""
        if self.backend == "flat":
            k = min(k, len(self.vecs))
            sims = q @ self.vecs.T
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            order = np.take_along_axis(sims, top, 1).argsort(axis=1)[:, ::-1]
            return np.take_along_axis(top, order, 1)
        _, idx = self.faiss.search(q, k)
        return idx


class BM25Index:
    def __init__(self, texts: list[str]):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([self._tok(t) for t in texts])

    @staticmethod
    def _tok(t: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", t.lower())

    def search(self, queries: list[str], k: int) -> np.ndarray:
        return np.stack([np.argsort(-self.bm25.get_scores(self._tok(q)))[:k] for q in queries])


def rrf(rankings: list[np.ndarray], k: int, c: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of several (n_queries, depth) rankings."""
    out = []
    for per_query in zip(*rankings):
        scores: dict[int, float] = {}
        for ranked in per_query:
            for r, idx in enumerate(ranked):
                if idx >= 0:
                    scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (c + r + 1)
        out.append(sorted(scores, key=scores.get, reverse=True)[:k])
    return np.array([row + [-1] * (k - len(row)) for row in out])
