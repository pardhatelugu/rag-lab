# Step 1: Embeddings and similarity

**Goal:** implement `cosine_matrix` and `top_k` in `src/ragbench/similarity.py`.
**Check:** `pytest tests/test_step1_similarity.py`

## Concepts (10 min)
- An **embedding** maps text to a vector so that similar meaning means nearby vectors. Retrieval is then a nearest-neighbour problem.
- **Cosine similarity** compares direction, not length: `cos(a,b) = a·b / (|a||b|)`. If vectors are L2-normalized, cosine equals the dot product. That is why embedding libraries normalize and why vector DBs offer "inner product" as a metric.
- Scoring N queries against M chunks is one matrix multiply. Never loop in Python.
- `top_k` only needs the k best, not a full sort. `argpartition` is O(n); sorting is O(n log n).

## Do it
1. Implement the two functions.
2. In a scratch notebook, embed 5 sentences with `ragbench.embed.STEmbedder("BAAI/bge-small-en-v1.5")`: two about cooking, two about neural nets, one ambiguous ("Python is great"). Print the similarity matrix. Which pair surprises you?
3. Compare `HashEmbedder` vs `STEmbedder` on the same sentences. What does the hash embedder miss? (Synonyms. This is why dense retrieval exists.)

## Reflect (write 3 sentences in your repo's NOTES.md)
- Why do normalized embeddings make search cheaper?
- What happens to a zero vector, and why would an embedding model ever produce one?

**Commit:** `git commit -am "step 1: cosine similarity and top-k"`

---

# Step 2: Chunking

**Goal:** implement `fixed_chunks`, `sentence_chunks`, `recursive_chunks` (and `semantic_chunks` as a stretch after Step 3) in `chunkers.py`.
**Check:** `pytest tests/test_step2_chunking.py`

## Concepts (15 min)
- Embedding models have input limits and blur meaning when a chunk covers many topics. Too big dilutes the vector; too small loses context. Chunking is the first real tuning knob in RAG.
- **Fixed-size**: simple, ignores structure, cuts mid-sentence. Your baseline.
- **Sentence windows**: respects sentence boundaries, size varies.
- **Recursive**: tries paragraph, then line, then sentence, then word boundaries. Usually the best default.
- **Semantic**: cuts where topic shifts, found by a dip in similarity between adjacent sentences. Costs an embedding call per sentence.
- **Overlap** reduces the chance that an answer is split across a boundary, at the cost of more chunks.
- **Offsets matter.** Keeping `(start, end)` into the original text is what lets Step 4 grade retrieval objectively.

## Do it
Implement in this order: fixed (easiest), sentence, recursive. Run the tests after each one.

## Reflect
- Print the first 3 chunks from each strategy on a real article. Which ones read like something a human would quote?
- Predict: which strategy will win in Step 4, and why? Write your guess in NOTES.md now, before you see the numbers.

**Commit** after each chunker: `step 2a: fixed chunks`, `2b: sentence`, `2c: recursive`.

---

# Step 3: Indexing

**Goal:** implement `VectorIndex` (flat and HNSW) in `index.py`.
**Check:** `pytest tests/test_step3_index.py` (install `faiss-cpu` for the HNSW test)

## Concepts (15 min)
- **Flat (exact)** search compares the query to every vector. Perfect recall, O(N) per query. Fine up to roughly a million vectors.
- **ANN (approximate)** indexes trade a little recall for big speed. **HNSW** builds a layered proximity graph: coarse layers jump far, fine layers refine. Knobs: `M` (links per node; more means better recall and more memory) and `efSearch` (candidates explored per query; more means better recall and slower).
- Other families you should be able to name: IVF (cluster then search nearest clusters), PQ (compress vectors), DiskANN. Vector DBs (Qdrant, Chroma, pgvector) wrap these.

## Do it
1. Implement flat using Step 1, then HNSW with `faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)`.
2. Experiment: generate 200k random normalized vectors, time flat vs HNSW, and compute HNSW recall against flat as ground truth. Sweep `efSearch` in `[8, 32, 128]` and plot the recall-vs-latency curve. Save the plot in `docs/`. This one chart is excellent portfolio material.

## Reflect
- At what corpus size would you stop using flat search? What would you check first?

**Commit:** `step 3: flat and HNSW index + ef_search sweep`

---

# Step 4: Evaluation

**Goal:** implement `relevant_sets` and `score` in `evaluate.py`.
**Check:** `pytest tests/test_step4_eval.py`

## Concepts (10 min)
- You cannot improve retrieval you cannot measure. A retrieval eval needs queries plus known-relevant chunks.
- **Recall@k (hit rate):** did at least one relevant chunk appear in the top k? **MRR:** how high did the first relevant chunk rank? Averaging 1/rank rewards putting the answer first.
- Our ground truth is a proxy: `"Title: Section"` queries, with relevance from section offsets. It is free and exact, but easy. Know that limitation and say so in your README; reviewers respect that.

## Do it
Implement both functions, then run the full benchmark with `--model hash` to confirm the pipeline works end to end, and again with the real model.

**Commit:** `step 4: evaluation metrics`

---

# Step 5: Hybrid search

**Goal:** implement `BM25Index` and `rrf` in `index.py`.
**Check:** `pytest tests/test_step5_hybrid.py`

## Concepts (10 min)
- Dense embeddings capture meaning but can miss exact terms (names, IDs, rare acronyms). **BM25** is a keyword scorer (TF-IDF with length normalization and saturation) that nails exact matches.
- **Hybrid** runs both and merges. Raw scores are on different scales, so **Reciprocal Rank Fusion** merges by rank: `score = sum(1 / (c + rank))`. No tuning of score weights needed.

## Do it
Implement both, then run the full benchmark. Compare dense vs hybrid per chunker.

**Commit:** `step 5: BM25 and RRF hybrid`

---

# Step 6: Run it, analyze it, ship it

1. `python -m ragbench.run_benchmark` and save `results/results.md`.
2. Write the **Results** section of your README: a table, one chart, and 4 sentences: which strategy won, whether your Step 2 prediction held, where hybrid helped, and one limitation.
3. Add the HNSW `efSearch` plot from Step 3.
4. Stretch goals: add a cross-encoder reranker, swap embedding models, generate LLM questions for a harder eval.

Your git log (`step 1` through `step 6`) shows recruiters how you work, not just what you built.
