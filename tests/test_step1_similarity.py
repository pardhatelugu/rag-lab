import numpy as np

from ragbench.similarity import cosine_matrix, top_k


def test_cosine_basic():
    a = np.array([[1.0, 0], [0, 2.0]])
    b = np.array([[3.0, 0], [0, 1.0], [1.0, 1.0]])
    s = cosine_matrix(a, b)
    assert s.shape == (2, 3)
    assert np.allclose(s[0], [1, 0, 1 / np.sqrt(2)])
    assert np.allclose(s[1], [0, 1, 1 / np.sqrt(2)])


def test_cosine_zero_vector_is_safe():
    s = cosine_matrix(np.zeros((1, 3)), np.ones((2, 3)))
    assert np.all(np.isfinite(s)) and np.allclose(s, 0)


def test_top_k_order_and_clip():
    sims = np.array([[0.1, 0.9, 0.5, 0.7], [0.3, 0.2, 0.8, 0.1]])
    assert top_k(sims, 2).tolist() == [[1, 3], [2, 0]]
    assert top_k(sims, 99).shape == (2, 4)
