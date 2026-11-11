import numpy as np
import pytest

from ragbench.embed import HashEmbedder
from ragbench.index import VectorIndex


def test_flat_finds_itself():
    v = HashEmbedder().encode(["alpha beta", "gamma delta", "epsilon zeta"])
    idx = VectorIndex("flat")
    idx.add(v)
    assert idx.search(v, 1)[:, 0].tolist() == [0, 1, 2]


def test_flat_k_larger_than_corpus():
    v = HashEmbedder().encode(["a b", "c d"])
    idx = VectorIndex("flat")
    idx.add(v)
    assert idx.search(v, 10).shape == (2, 2)


def test_hnsw_matches_flat_on_small_data():
    faiss = pytest.importorskip("faiss")  # noqa: F841
    rng = np.random.default_rng(0)
    v = rng.normal(size=(300, 32)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[:20]
    flat, hnsw = VectorIndex("flat"), VectorIndex("hnsw")
    flat.add(v)
    hnsw.add(v)
    a, b = flat.search(q, 5), hnsw.search(q, 5)
    recall = np.mean([len(set(x) & set(y)) / 5 for x, y in zip(a, b)])
    assert recall > 0.9
