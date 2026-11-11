"""Evaluation: build (query -> relevant chunks) ground truth from section
boundaries and compute Recall@k / MRR / latency for a chunker + retriever."""
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
    """Proxy benchmark: query = '<article title>: <section heading>', gold = any
    chunk lying mostly inside that section. Cheap and free, but headings are an
    easy signal; swap in LLM-written questions for a harder eval."""
    qs = []
    for d in docs:
        for s in d.sections:
            if s.end - s.start >= MIN_SECTION_CHARS:
                qs.append(Query(f"{d.title}: {s.heading}", d.id, s.start, s.end))
    return qs


def relevant_sets(chunks: list[Chunk], queries: list[Query], min_frac: float = 0.5) -> list[set[int]]:
    rel = []
    for q in queries:
        hits = set()
        for i, c in enumerate(chunks):
            if c.doc_id != q.doc_id:
                continue
            overlap = min(c.end, q.end) - max(c.start, q.start)
            if overlap > 0 and overlap / max(1, c.end - c.start) >= min_frac:
                hits.add(i)
        rel.append(hits)
    return rel


def score(ranked: np.ndarray, rel: list[set[int]], ks=(1, 3, 5, 10)) -> dict[str, float]:
    keep = [i for i, r in enumerate(rel) if r]  # skip queries with no gold chunk
    out = {f"recall@{k}": float(np.mean([bool(set(ranked[i][:k].tolist()) & rel[i]) for i in keep])) for k in ks}
    rr = []
    for i in keep:
        pos = [r for r, idx in enumerate(ranked[i]) if int(idx) in rel[i]]
        rr.append(1.0 / (pos[0] + 1) if pos else 0.0)
    out["mrr"] = float(np.mean(rr))
    out["n_queries"] = len(keep)
    return out


def run_config(docs, chunker, embedder, *, backend="flat", mode="dense", depth=20) -> dict:
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
