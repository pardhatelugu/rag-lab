"""CLI: python -m ragbench.run_benchmark --model BAAI/bge-small-en-v1.5"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .chunkers import get_chunker
from .data import load_corpus
from .embed import HashEmbedder, STEmbedder
from .evaluate import run_config

CONFIGS = [
    ("fixed", {"size": 500, "overlap": 50}),
    ("fixed", {"size": 1000, "overlap": 100}),
    ("recursive", {"size": 500, "overlap": 50}),
    ("recursive", {"size": 1000, "overlap": 100}),
    ("sentence", {"n": 5, "stride": 4}),
    ("semantic", {"percentile": 25.0}),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="sentence-transformers model, or 'hash' for offline")
    p.add_argument("--backend", default="flat", choices=["flat", "hnsw"])
    p.add_argument("--out", default="results")
    args = p.parse_args()

    docs = load_corpus()
    embedder = HashEmbedder() if args.model == "hash" else STEmbedder(args.model)
    print(f"{len(docs)} docs, model={embedder.name}")

    rows = []
    for name, kw in CONFIGS:
        chunker = get_chunker(name, embedder, **kw)
        for mode in ("dense", "hybrid"):
            r = run_config(docs, chunker, embedder, backend=args.backend, mode=mode)
            row = {"chunker": name, "params": str(kw), "mode": mode, **r}
            rows.append(row)
            print(f"{name:9} {str(kw):34} {mode:6} R@5={r['recall@5']:.3f} MRR={r['mrr']:.3f} chunks={r['n_chunks']}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["chunker", "params", "mode", "recall@1", "recall@5", "recall@10", "mrr", "n_chunks", "search_ms_per_query"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (out / "results.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/results.csv and results.md")


if __name__ == "__main__":
    main()
