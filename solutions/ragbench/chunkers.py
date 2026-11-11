"""Chunking strategies. Every chunk keeps character offsets into its source
document so retrieval can be evaluated against ground-truth spans."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str


def _make(doc_id: str, text: str, spans: list[tuple[int, int]]) -> list[Chunk]:
    out = []
    for s, e in spans:
        t = text[s:e].strip()
        if t:
            out.append(Chunk(doc_id, s, e, t))
    return out


def fixed_chunks(doc_id: str, text: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Naive sliding window over characters. The baseline everyone starts with."""
    step = max(1, size - overlap)
    spans = [(s, min(s + size, len(text))) for s in range(0, len(text), step)]
    return _make(doc_id, text, spans)


def _split_spans(text, start, end, size, seps):
    if end - start <= size:
        return [(start, end)]
    if not seps:
        return [(s, min(s + size, end)) for s in range(start, end, size)]
    sep, pieces, pos = seps[0], [], start
    while pos < end:
        i = text.find(sep, pos, end)
        stop = end if i == -1 else i + len(sep)
        pieces.append((pos, stop))
        pos = stop
    out = []
    for s, e in pieces:
        out.extend(_split_spans(text, s, e, size, seps[1:]) if e - s > size else [(s, e)])
    return out


def recursive_chunks(doc_id: str, text: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split on paragraph, then line, then sentence, then word boundaries, and
    merge neighbours up to `size`. Overlap is added by extending each chunk
    backwards to a word boundary."""
    if not text.strip():
        return []
    pieces = _split_spans(text, 0, len(text), size, ["\n\n", "\n", ". ", " "])
    merged, cs, ce = [], *pieces[0]
    for s, e in pieces[1:]:
        if e - cs <= size:
            ce = e
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    spans = []
    for i, (s, e) in enumerate(merged):
        if i > 0 and overlap:
            back = max(0, s - overlap)
            sp = text.find(" ", back, s)
            s = sp + 1 if sp != -1 else s
        spans.append((s, e))
    return _make(doc_id, text, spans)


_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)\s*")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _SENT.finditer(text) if m.group().strip()]


def sentence_chunks(doc_id: str, text: str, n: int = 5, stride: int = 4) -> list[Chunk]:
    """Windows of `n` sentences, advancing `stride` sentences (overlap = n - stride)."""
    sents = _sentence_spans(text)
    spans = [(sents[i][0], sents[min(i + n, len(sents)) - 1][1]) for i in range(0, len(sents), stride)]
    return _make(doc_id, text, spans)


def semantic_chunks(doc_id, text, embedder, percentile: float = 25.0, max_chars: int = 1500, min_chars: int = 200):
    """Embed sentences and cut where adjacent-sentence similarity drops into the
    lowest `percentile` of the document, subject to min/max chunk size."""
    sents = _sentence_spans(text)
    if len(sents) < 3:
        return _make(doc_id, text, [(0, len(text))])
    vecs = embedder.encode([text[s:e] for s, e in sents])
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    cut_below = np.percentile(sims, percentile)
    spans, cs = [], sents[0][0]
    for i, (s, e) in enumerate(sents):
        size = e - cs
        is_last = i == len(sents) - 1
        boundary = (not is_last) and sims[i] <= cut_below and size >= min_chars
        if is_last or boundary or size >= max_chars:
            spans.append((cs, e))
            if not is_last:
                cs = sents[i + 1][0]
    return _make(doc_id, text, spans)


def get_chunker(name: str, embedder=None, **kw):
    """Return f(doc_id, text) -> list[Chunk] for a strategy name."""
    table = {"fixed": fixed_chunks, "recursive": recursive_chunks, "sentence": sentence_chunks}
    if name in table:
        return lambda d, t: table[name](d, t, **kw)
    if name == "semantic":
        return lambda d, t: semantic_chunks(d, t, embedder, **kw)
    raise ValueError(f"unknown chunker {name}")
