"""Embedders. `HashEmbedder` is deterministic and offline (used in tests/CI);
`STEmbedder` wraps any open-source sentence-transformers model."""
from __future__ import annotations

import hashlib
import re

import numpy as np


class HashEmbedder:
    """Bag-of-words hashing trick. Not semantic, but fast and dependency-free."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)


class STEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 query_prefix: str = "Represent this sentence for searching relevant passages: "):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64,
                                 show_progress_bar=False).astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode([self.query_prefix + t for t in texts])
