# Chunking benchmark (`bench/`)

Compares two chunking methods — **maruf** vs **wildan** (folders under
`chunks_data/`) — to decide which produces a better **knowledge graph**. "Better
for a KG" is not the same as "better for RAG": extraction runs **per chunk**, so
the winning method is the one that keeps each entity together with its relations
inside one chunk. Split them across a boundary and the edge is lost.

## The 3 layers of evidence

| Layer | Script | Needs | Answers |
|---|---|---|---|
| 1. Intrinsic | `chunk_metrics.py` | nothing (stdlib) | Do the chunks *look* right — sized well, cut at sentence boundaries, not fragmented? |
| 2. Graph structure | `graph_metrics.py` | Ollama LLM (`.env`) | Does the extracted graph come out connected, low-duplication, well-linked? |
| 3. Task recall | `eval_gold.py` | layer 2 output + gold set | Did it actually recover the entities/relations we KNOW are in the doc? |

Neither Neo4j nor embeddings are needed — the graph is built in-memory with
networkx. Only layer 2/3 need the `RE_MODEL_*` settings already in `.env`.

## Run order

```bash
# 0. normalize both methods' native formats -> one loadable .jsonl schema
python bench/prep.py

# 1. intrinsic metrics (instant, no API calls)
python bench/chunk_metrics.py

# 2. build + measure the graph per method (calls the LLM; --limit for a smoke test)
python bench/graph_metrics.py --limit 20      # quick, first 20 chunks
python bench/graph_metrics.py                 # full run

# 3. score extracted graphs against the gold set
python bench/eval_gold.py -v
```

The default document set is the 2 files chosen for experiment #1 (present in both
methods): `Thesis_M10801107_Yu_Ting_Chiu`, `Thesis_M11001010_Hung_Chun_Tse_Nick`.
Override with `--doc <name>` (repeatable).

## Metric cheat-sheet

**Intrinsic** — `ends_sent%` is the headline: fraction of chunks ending on a
sentence boundary. Mid-sentence cuts are exactly where cross-chunk relations get
lost, so higher is better. Also watch chunk count (more = more fragmentation) and
`tiny`/`huge` outliers.

**Graph** — `lcc_frac` (largest connected component / all nodes; higher = one
coherent graph, not islands) is the headline. `dup` ≈ how often the same entity
was re-emitted across chunks and merged back (≈1.0 ideal; high = fragmentation).
`isolated%`/`degree1%` = weakly-linked nodes (lower better). Read modularity
*together with* `lcc_frac` — high modularity + low lcc means fragmented, not good.

**Recall** — `topic_recall` / `agent_recall` against the gold set. Precision is
intentionally not scored (gold set is abstract-derived, not exhaustive).

## Fairness notes

- `prep.py` applies the SAME cleaning to both methods (strip markup-only chunks
  like `<!-- image -->`, drop sub-`MIN_CHARS` fragments) and reports how many each
  lost. Method-specific flags (maruf's `extraction_eligible`) are ignored by
  default; `--respect-eligible` honors them.
- Gold sets in `gold/*.json` are **DRAFTS derived from the English abstract** and
  are marked as needing human review. Treat recall numbers as directional until
  the gold set is signed off.
