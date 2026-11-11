"""STEP 4: evaluation. Implement `relevant_sets` and `score`; the rest is provided.

Ground truth: query = '<article title>: <section heading>'. A chunk is relevant if at
least `min_frac` of the chunk lies inside that section's character range.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .chunkers import Chunk
from .data import Document
from .index import BM25Index, VectorIndex, rrf

MIN_SECTION_CHARS = 300


@dataclass
class Query:
    text: str
    doc_id: str
    start: int
    end: int


def build_queries(docs: list[Document]) -> list[Query]:
    """Provided."""
    qs = []
    for d in docs:
        for s in d.sections:
            if s.end - s.start >= MIN_SECTION_CHARS:
                qs.append(Query(f"{d.title}: {s.heading}", d.id, s.start, s.end))
    return qs


def relevant_sets(chunks: list[Chunk], queries: list[Query], min_frac: float = 0.5) -> list[set[int]]:
    """4a. For each query, return the set of chunk indices that are relevant.

    Only chunks with the same doc_id can match. overlap = min(c.end, q.end) - max(c.start, q.start).
    Relevant if overlap > 0 and overlap / chunk_length >= min_frac.
    Think: why divide by the chunk length and not the section length?
    """
    raise NotImplementedError


def score(ranked: np.ndarray, rel: list[set[int]], ks=(1, 3, 5, 10)) -> dict[str, float]:
    """4b. `ranked` is (n_queries, depth) chunk indices, best first.

    Return {'recall@k': ..., 'mrr': ..., 'n_queries': ...}.
      recall@k = fraction of queries with >= 1 relevant chunk in the top k
      mrr      = mean of 1/(rank of first relevant chunk), 0 if none retrieved
    Skip queries whose relevant set is empty (the chunker lost that section).
    Think: why is recall@k a hit-rate here rather than a fraction of relevant chunks found?
    """
    raise NotImplementedError


def run_config(docs, chunker, embedder, *, backend="flat", mode="dense", depth=20) -> dict:
    """Provided: chunk -> embed -> index -> search -> score."""
    chunks = [c for d in docs for c in chunker(d.id, d.text)]
    texts = [c.text for c in chunks]
    queries = build_queries(docs)
    rel = relevant_sets(chunks, queries)

    t0 = time.perf_counter()
    vecs = embedder.encode(texts)
    embed_s = time.perf_counter() - t0

    index = VectorIndex(backend)
    index.add(vecs)
    qtexts = [q.text for q in queries]
    qvecs = embedder.encode_queries(qtexts)

    t0 = time.perf_counter()
    dense = index.search(qvecs, depth)
    if mode == "dense":
        ranked = dense
    else:
        ranked = rrf([dense, BM25Index(texts).search(qtexts, depth)], depth)
    search_ms = (time.perf_counter() - t0) / max(1, len(queries)) * 1000

    res = score(ranked, rel)
    res.update(n_chunks=len(chunks), avg_chunk_chars=float(np.mean([len(t) for t in texts])),
               embed_seconds=round(embed_s, 2), search_ms_per_query=round(search_ms, 3))
    return res
