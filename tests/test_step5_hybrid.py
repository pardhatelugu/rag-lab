import numpy as np

from ragbench.chunkers import get_chunker
from ragbench.data import Document, parse_sections
from ragbench.embed import HashEmbedder
from ragbench.evaluate import run_config
from ragbench.index import BM25Index, rrf


def test_bm25_ranks_keyword_match_first():
    texts = ["the cat sat on the mat", "graph based nearest neighbour search", "stock market news today"]
    idx = BM25Index(texts)
    assert idx.search(["nearest neighbour graph"], 2)[0][0] == 1


def test_rrf_rewards_agreement():
    a = np.array([[1, 2, 3]])
    b = np.array([[2, 1, 4]])
    out = rrf([a, b], 4)[0].tolist()
    assert set(out[:2]) == {1, 2} and set(out[2:]) == {3, 4}


def test_rrf_pads_with_minus_one():
    out = rrf([np.array([[7]])], 3)[0].tolist()
    assert out == [7, -1, -1]


def test_rrf_exact_scores():
    # item 0: rank0 in both -> 2/61 ; item 1: rank1 in one -> 1/62
    out = rrf([np.array([[0, 1]]), np.array([[0, 2]])], 3)[0].tolist()
    assert out[0] == 0


def test_end_to_end_hybrid():
    text = "Intro text. " * 10 + "\n== Indexing ==\n" + "HNSW graph search. " * 40 + "\n== Chunking ==\n" + "Split on paragraphs. " * 40
    docs = [Document(f"d{i}", f"Doc {i}", text, parse_sections(text)) for i in range(3)]
    for mode in ("dense", "hybrid"):
        r = run_config(docs, get_chunker("recursive", size=500, overlap=50), HashEmbedder(), mode=mode)
        assert r["mrr"] > 0.3 and r["recall@10"] >= r["recall@1"]
