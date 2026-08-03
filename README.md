# Knowledge Graph Builder

A minimal, script-only pipeline that turns a **pre-built chunks dataset** into a
**Knowledge Graph stored in Neo4j**, using an LLM to extract entities and
relationships from each chunk.

It is a stripped-down version of a larger GraphRAG project: there is **no
Streamlit UI, no document loading/cleaning/chunking, and no retrieval/Q&A**.
The chunks already exist (as `.jsonl` files in [chunks_data/](chunks_data/)),
so this repo only covers the path **from chunks to a built graph**.

## Pipeline

```
load chunks (.jsonl)  ->  embed  ->  extract graph (LLM)  ->  store in Neo4j
                                                                 -> centralities & communities
```

1. **Load** — read pre-built chunks from `.jsonl` files ([ChunksIngestor](src/ingestion/chunks_ingestor.py)).
2. **Embed** — vectorize each chunk ([ChunkEmbedder](src/ingestion/embedder.py)).
3. **Extract** — an LLM turns each chunk into nodes + relationships against a
   **fixed ontology** (the LLM fills in instances, it does not define the schema),
   then a deterministic sanitizer enforces that ontology — fixing relationship
   directions, collapsing duplicate / near-duplicate nodes across chunks, capping
   `has_source`, and dropping anything off-ontology
   ([GraphMiner](src/ingestion/graph_miner.py) → [GraphExtractor](src/agents/graph_extractor.py) → [sanitize_graph](src/graph/graph_model.py)).
4. **Store** — write `Document`, `Chunk` (with embeddings), entity nodes and
   `PART_OF` / `NEXT` / `MENTIONS` relationships to Neo4j
   ([KnowledgeGraph](src/graph/knowledge_graph.py)).
5. **Enrich** — compute PageRank / betweenness / closeness and detect
   Leiden & Louvain communities ([graph_ds](src/graph/graph_ds.py)).

## Ontology

The ontology is **fixed and defined in this repo** — not something the LLM invents.
It lives in the extraction prompt ([prompts/graph_extractor.py](src/prompts/graph_extractor.py))
and is re-enforced deterministically by [sanitize_graph](src/graph/graph_model.py),
so the LLM only decides *which* instances to extract, never the schema.

**Node types extracted per-chunk (6):** `Agent`, `Role`, `Topic`, `Type`, `Source`,
`Description`. A 7th label, `Contradiction`, exists in the ontology but is no
longer emitted here — construction-time contradiction detection (formerly
STEP D) was disabled; `Contradiction` is now produced exclusively by the
separate on-demand pass — see [Conflict ontology](#conflict-ontology-contradiction--supersedes) below.

**Relationships (extracted from text):**

| Relationship             | Direction        | Notes |
|--------------------------|------------------|-------|
| `role_in_meeting` / `role_in_paper` | Agent → Role | |
| `spoke_about` / `writes_about`      | Agent → Topic | |
| `has_source`             | Topic → Source   | only the 1–3 top-level Topics; capped per document |
| `has_[type]`             | Topic → Type     | e.g. `has_method`, `has_research_problem` |
| `has_[type]_description` | Type → Description| |
| `has_subtopic`           | Topic → Topic    | broader → narrower |
| `relates_to`             | Topic → Topic    | needs a `relation` from a controlled vocab (`addresses`, `resolves`, `produces`, `evaluates`, `follows_up_on`, `motivates`, `contradicts`, `identifies`) |
| `assigned_to`            | Topic → Agent    | only when the Topic's Type is `Action Item` |

`has_contradiction` (Description → Contradiction) is **not** in the table above
— it's no longer extracted per-chunk. See [Conflict ontology](#conflict-ontology-contradiction--supersedes) below for where it now comes from.

What `sanitize_graph` guarantees deterministically (regardless of what the model
emits): one canonical `Source` node per document, correct relationship directions,
no self-loops, snake_case relationship names, bare-abbreviation and near-duplicate
Topics merged, each `Type` tagged with a `domain` (`paper` / `meeting` / `shared`),
and `relates_to` / `assigned_to` dropped unless they satisfy the rules above.

Separately, the graph store adds **structural** relationships (not LLM-extracted):
`PART_OF` and `NEXT` (Chunk-level), `MENTIONS` (Chunk → entity), and — opt-in via
`create_precedes_relationships`, requiring `series` + `date` metadata — `PRECEDES`
(Document → Document).

### Conflict ontology (Contradiction & supersedes)

> **Superseded design below.** This section (and `docs/conflict_ontology.md`,
> which it was based on) describes the pre-`conflict_pipeline.md` schema —
> e.g. `Contradiction`'s `level` property no longer exists, replaced by
> `resolution_type`/`confidence`/`participants_hash`/etc. Current authoritative
> spec: [docs/conflict_pipeline.md](docs/conflict_pipeline.md). What stays true
> either way: construction-time (per-chunk) contradiction detection is
> disabled — `Contradiction`/`has_contradiction` are produced only by the
> separate on-demand pass described below in outline.

Beyond construction, a **separate, on-demand pass** scans the whole graph for
conflicting facts and classifies each one. It runs **only when called**, and
because it sees the entire KB (not one chunk) it catches cross-chunk /
cross-document conflicts that per-chunk extraction cannot. Each conflict resolves
to **one of two** shapes:

| Situation | Output | New node? |
|-----------|--------|-----------|
| Two facts genuinely conflict and both still stand | `Contradiction` node + `has_contradiction` edges | yes |
| A newer `Result` corrects / replaces an older one | `supersedes` edge | no — edge only |

- **`Contradiction`** — a reified conflict with a required `summary` (must name
  the specific clashing detail from both sides). Anchors **≥2** `has_contradiction`
  edges (`Description → Contradiction`, each with a `level`); a singleton is
  meaningless and is removed downstream
  ([`_cleanup_singleton_contradictions`](src/graph/knowledge_graph.py)).
- **`supersedes`** — `Description → Description`, **both `typeName = "Result"`**,
  direction newer → older, optional `reason`, **no node** (it just records that
  the new fact updates the old). Its constraints (both endpoints `Result`, no
  self-loop, no `A↔B` cycle) are enforced by the detection pass, not the per-chunk
  sanitizer.

**Decision rule:** if the newer Result *corrects* the older one → `supersedes`
edge; if both genuinely stand in opposition → `Contradiction` node.

The node/edge shapes referenced above are historical (`graph_model.py` still
carries the old constants, e.g. `ALLOWED_CONTRADICTION_LEVEL`, unchanged and
inert). Full current spec — decision rules, output ontology, constraints,
example Cypher — in [docs/conflict_pipeline.md](docs/conflict_pipeline.md).

## Requirements

- **Neo4j** — a running instance (local desktop or a free Neo4j Aura cloud instance).
- **An LLM for extraction** — configured for [Groq](https://console.groq.com/)
  cloud by default (`llama-3.3-70b-versatile`). No local GPU needed.
- **An embeddings model** — [Ollama](https://ollama.com/) `mxbai-embed-large` by
  default (small, runs on a laptop); can be swapped for OpenAI/Azure/HF.

## Setup

```bash
pip install -r requirements.txt

cp .env_example .env
# then edit .env with your Neo4j credentials and Groq API key
# (SUPABASE_* only needed if you pull chunks from a producer — see below)
```

If you use the default Ollama embedder, pull the model once:

```bash
ollama pull mxbai-embed-large
```

## Run

```bash
# ingest everything under ./chunks_data
python main.py

# quick test on the first 2 documents only
python main.py --limit 2

# point at a specific file or folder
python main.py --chunks path/to/chunks.jsonl

# skip the centralities/community-detection step
python main.py --no-communities
```

## Chunks format

Each line of a `.jsonl` file is one chunk. Full contract:
[docs/chunk_schema.md](docs/chunk_schema.md) — enforced by
`src/ingestion/chunk_record.py`.

| Field | Required | Description |
|---|---|---|
| `text` | ✅ | Chunk text |
| `doc_id` | ✅ | Document identifier (groups chunks) |
| `index` | strongly recommended | Integer position in the doc. Drives `NEXT` edges — **without it a document silently gets none** |
| `source_path`, `source_kind`, `n_chunks` | optional | Stored as `Document` metadata |
| `chunk_id` | optional | Producer's own id; only a fallback when `index` is missing |

`source_file` / `source_type` are accepted as aliases for `source_path` /
`source_kind`. Unrecognised fields are allowed and reported, never silently used.

Validate a file before spending hours extracting from it:

```bash
python -m src.ingestion.validate chunks_data/maruf
```

Exit code 1 on errors (invalid JSON, missing `text`, duplicate chunk ids);
warnings cover the quieter problems — gapped `index`, missing `source_kind`,
`n_chunks` disagreeing with the real count.

## Chunk hand-off (Supabase)

Chunk producers upload files to Supabase Storage and register them in a
`chunk_uploads` table; this machine polls and pulls. Setup, credentials and the
design rationale: [docs/chunk_sync.md](docs/chunk_sync.md).

```bash
# producer side — validates first, and one bad file uploads nothing
python scripts/upload_chunks.py out/*.jsonl

# this side
python run_sync.py                        # pull everything pending
python run_sync.py --dry-run              # verify + validate, write nothing
python run_sync.py --mark-built <doc_id>  # after main.py has built it
```

Each pull re-checks the SHA-256 and re-runs the validator before the file is
written; a failure lands in the row's `error` column where the producer can read
it.

A launchd agent runs the pull every 10 minutes, logging to
`~/Library/Logs/chunksync.log`:

```bash
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.kg.chunksync.plist  # install
launchctl bootout   gui/$UID/com.kg.chunksync                               # remove
```

It **only downloads**. Builds stay manual on purpose: two uploads arriving
minutes apart would otherwise start two multi-hour extraction runs competing for
the same Ollama and Neo4j.

## Layout

```
main.py                     # entry point (CLI)
run_sync.py                 # pull chunk uploads from Supabase
chunks_data/                # pre-built .jsonl chunks
db/supabase_schema.sql      # chunk_uploads table, bucket, RLS policies
scripts/upload_chunks.py    # producer-side uploader (run on their machine)
src/
  config.py                 # pydantic configuration models
  schema.py                 # Chunk / ProcessedDocument
  factory/                  # LLM + embeddings factories
  ingestion/                # chunks loader, embedder, graph miner
    chunk_record.py         # the .jsonl field contract (aliases, defaults)
    validate.py             # `python -m src.ingestion.validate <path>`
  sync/supabase_sync.py     # download, checksum, validate, mark status
  agents/graph_extractor.py # LLM agent that extracts the graph
  prompts/graph_extractor.py# extraction prompt (fixed ontology, hardcoded — not configurable)
  graph/                    # graph model, Neo4j store, graph-DS metrics
  utils/logger.py
```
