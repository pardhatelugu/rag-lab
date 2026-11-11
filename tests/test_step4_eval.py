import numpy as np

from ragbench.chunkers import Chunk
from ragbench.evaluate import Query, relevant_sets, score


def test_relevant_sets_overlap_rule():
    chunks = [Chunk("a", 0, 100, "x"), Chunk("a", 100, 200, "y"), Chunk("a", 150, 400, "z"), Chunk("b", 100, 200, "w")]
    q = [Query("q", "a", 100, 300)]
    # chunk0 no overlap; chunk1 fully inside; chunk2 overlap 150/250=0.6; chunk3 wrong doc
    assert relevant_sets(chunks, q) == [{1, 2}]


def test_relevant_sets_min_frac():
    chunks = [Chunk("a", 0, 100, "x")]
    assert relevant_sets(chunks, [Query("q", "a", 80, 500)]) == [set()]  # only 20% inside


def test_score_hit_rate_and_mrr():
    ranked = np.array([[5, 1, 2], [9, 8, 7], [3, 4, 0]])
    rel = [{1}, {0}, {0}]
    s = score(ranked, rel, ks=(1, 3))
    assert s["recall@1"] == 0.0
    assert s["recall@3"] == 2 / 3
    assert np.isclose(s["mrr"], (1 / 2 + 0 + 1 / 3) / 3)
    assert s["n_queries"] == 3


def test_score_skips_empty_gold():
    ranked = np.array([[1, 2], [3, 4]])
    s = score(ranked, [{1}, set()], ks=(1,))
    assert s["n_queries"] == 1 and s["recall@1"] == 1.0
