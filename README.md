# RAG Lab: build a retrieval pipeline step by step

A hands-on course in embeddings, chunking, indexing and hybrid search. You implement each piece, tests tell you when it is right, and you finish with a benchmark you can show recruiters.

**Free stack:** Wikipedia API (data), sentence-transformers (open-source embeddings), FAISS (HNSW index), rank-bm25.

## How it works

1. Read the step in `lessons/LESSONS.md` (about 10-15 minutes of concepts).
2. Open the stub file; every `raise NotImplementedError` is yours to fill in.
3. Run that step's test until it is green.
4. Commit with the message suggested in the lesson.
5. Write your reflection in `NOTES.md`.

| Step | You build | File | Test |
|---|---|---|---|
| 1 | cosine similarity, top-k | `similarity.py` | `test_step1_similarity.py` |
| 2 | fixed / sentence / recursive chunkers | `chunkers.py` | `test_step2_chunking.py` |
| 3 | flat and HNSW vector index | `index.py` | `test_step3_index.py` |
| 4 | Recall@k and MRR | `evaluate.py` | `test_step4_eval.py` |
| 5 | BM25 and RRF hybrid | `index.py` | `test_step5_hybrid.py` |
| 6 | run, analyze, write up | `run_benchmark.py` | n/a |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git init && git add . && git commit -m "scaffold"
pytest                       # everything fails at first; that is the point
pytest tests/test_step1_similarity.py
```

## Rules

- Try for at least 20 minutes before opening `solutions/`.
- To check the answer key, run `PYTHONPATH=solutions pytest`. Or ask Claude for a hint instead of the answer.
- Delete `solutions/` before publishing if you want the repo to read as your own work.

## Final deliverable

When all steps pass, run `python -m ragbench.run_benchmark`, then add a Results section to this README with your table, plot and analysis.
