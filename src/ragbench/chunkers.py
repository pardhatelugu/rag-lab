"""STEP 2: chunking. Implement each strategy, one test at a time.

Every Chunk must keep (start, end) character offsets into the ORIGINAL text.
That is what lets us measure retrieval exactly in Step 4.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str


def _make(doc_id: str, text: str, spans: list[tuple[int, int]]) -> list[Chunk]:
    """Provided helper: turn (start, end) spans into Chunks, dropping blank ones."""
    out = []
    for s, e in spans:
        t = text[s:e].strip()
        if t:
            out.append(Chunk(doc_id, s, e, t))
    return out


def fixed_chunks(doc_id: str, text: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """2a. Sliding window over characters.

    Window `size`, advancing by size - overlap. Build spans, then return _make(...).
    Think: what goes wrong when a window cuts a word or sentence in half?
    """
    raise NotImplementedError


def sentence_chunks(doc_id: str, text: str, n: int = 5, stride: int = 4) -> list[Chunk]:
    """2b. Windows of `n` sentences advancing `stride` sentences (overlap = n - stride).

    First find sentence spans with a regex (finditer gives you offsets for free).
    A naive split on '.' breaks on 'e.g.' and 'U.S.'. Does it matter for retrieval?
    """
    raise NotImplementedError


def recursive_chunks(doc_id: str, text: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """2c. Split on the biggest separator that works: '\\n\\n', then '\\n', then '. ', then ' '.

    Algorithm sketch:
      1. If a span fits in `size`, keep it.
      2. Otherwise split it on the next separator (keep the separator attached) and recurse.
      3. Greedily merge neighbouring pieces while the merged span stays <= size.
      4. Add overlap by moving each chunk's start back ~`overlap` chars (snap to a space).
    Track offsets, not just strings. Use text.find(sep, pos, end).
    """
    raise NotImplementedError


def semantic_chunks(doc_id, text, embedder, percentile: float = 25.0, max_chars: int = 1500, min_chars: int = 200):
    """2d (stretch, after Step 3). Embed each sentence, then cut where the similarity
    between adjacent sentences falls into the lowest `percentile` of this document,
    subject to min/max chunk size. Reuse ragbench.similarity.
    """
    raise NotImplementedError


def get_chunker(name: str, embedder=None, **kw):
    """Provided: returns f(doc_id, text) -> list[Chunk]."""
    table = {"fixed": fixed_chunks, "recursive": recursive_chunks, "sentence": sentence_chunks}
    if name in table:
        return lambda d, t: table[name](d, t, **kw)
    if name == "semantic":
        return lambda d, t: semantic_chunks(d, t, embedder, **kw)
    raise ValueError(f"unknown chunker {name}")
