import pytest

from ragbench.chunkers import fixed_chunks, recursive_chunks, semantic_chunks, sentence_chunks
from ragbench.embed import HashEmbedder

TEXT = (
    "Intro paragraph about vectors. " * 5
    + "\n\nHNSW builds a layered graph for approximate nearest neighbour search. " * 15
    + "\n\nRecursive splitting respects paragraph and sentence boundaries. " * 15
)


def _valid(chunks):
    assert chunks
    for c in chunks:
        assert 0 <= c.start < c.end <= len(TEXT)
        assert c.text == TEXT[c.start:c.end].strip()


def test_fixed_offsets_and_size():
    ch = fixed_chunks("d", TEXT, size=300, overlap=50)
    _valid(ch)
    assert all(c.end - c.start <= 300 for c in ch)
    assert ch[1].start == 250  # step = size - overlap
    assert ch[-1].end == len(TEXT)


def test_sentence_windows():
    ch = sentence_chunks("d", TEXT, n=3, stride=2)
    _valid(ch)
    assert ch[0].text.count(".") == 3
    assert ch[1].start > ch[0].start and ch[1].start < ch[0].end  # overlapping windows


@pytest.mark.parametrize("size,overlap", [(300, 0), (500, 60)])
def test_recursive_covers_text(size, overlap):
    ch = recursive_chunks("d", TEXT, size=size, overlap=overlap)
    _valid(ch)
    assert min(c.start for c in ch) <= 5 and max(c.end for c in ch) >= len(TEXT) - 5
    assert all(c.end - c.start <= size + overlap + 20 for c in ch)


def test_recursive_prefers_paragraph_breaks():
    ch = recursive_chunks("d", TEXT, size=700, overlap=0)
    ends_clean = sum(TEXT[c.end - 1] in ".\n " for c in ch)
    assert ends_clean >= len(ch) - 1


def test_semantic_makes_multiple_chunks():
    assert len(semantic_chunks("d", TEXT, HashEmbedder(), min_chars=100)) >= 2
