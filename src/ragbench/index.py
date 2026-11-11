"""STEPS 3 and 5: indexing and hybrid search."""
from __future__ import annotations

import re

import numpy as np


class VectorIndex:
    """STEP 3. Two backends behind one interface.

    backend='flat': exact search. Brute-force similarity against every vector.
    backend='hnsw': approximate search with FAISS IndexHNSWFlat (inner product).
    Vectors are already L2-normalized, so inner product == cosine.
    """

    def __init__(self, backend: str = "flat", m: int = 32, ef_search: int = 64):
        self.backend, self.m, self.ef = backend, m, ef_search
        self.vecs = None
        self.faiss = None

    def add(self, vecs: np.ndarray) -> None:
        """Store the vectors (flat) or build the FAISS HNSW index (hnsw)."""
        raise NotImplementedError

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        """Return (n_queries, k) chunk indices, best first.
        Flat: reuse ragbench.similarity. HNSW: self.faiss.search(q, k) returns (scores, ids).
        """
        raise NotImplementedError


class BM25Index:
    """STEP 5a. Keyword search with rank_bm25.BM25Okapi.
    __init__: tokenize texts (lowercase, regex words) and build BM25Okapi.
    search: for each query, score all docs with get_scores and return top-k indices.
    """

    def __init__(self, texts: list[str]):
        raise NotImplementedError

    def search(self, queries: list[str], k: int) -> np.ndarray:
        raise NotImplementedError


def rrf(rankings: list[np.ndarray], k: int, c: int = 60) -> np.ndarray:
    """STEP 5b. Reciprocal Rank Fusion.

    `rankings` is a list of (n_queries, depth) index arrays, one per retriever.
    Each retriever gives item at rank r (0-based) a score of 1 / (c + r + 1); sum per item.
    Return (n_queries, k), best first, padded with -1 if fewer than k candidates.
    Why does RRF avoid having to normalize dense and BM25 scores onto one scale?
    """
    raise NotImplementedError
