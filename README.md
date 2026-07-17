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
3. **Extract** — an LLM turns each chunk into nodes + relationships, then a
   deterministic sanitizer enforces the ontology
   ([GraphMiner](src/ingestion/graph_miner.py) → [GraphExtractor](src/agents/graph_extractor.py) → [sanitize_graph](src/graph/graph_model.py)).
4. **Store** — write `Document`, `Chunk` (with embeddings), entity nodes and
   `PART_OF` / `NEXT` / `MENTIONS` relationships to Neo4j
   ([KnowledgeGraph](src/graph/knowledge_graph.py)).
5. **Enrich** — compute PageRank / betweenness / closeness and detect
   Leiden & Louvain communities ([graph_ds](src/graph/graph_ds.py)).

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

Each line of a `.jsonl` file is one chunk. Fields used by the loader:

| Field | Required | Description |
|---|---|---|
| `text` | ✅ | Chunk text |
| `doc_id` | ✅ | Document identifier (groups chunks) |
| `chunk_id` | ✅ | Original chunk identifier |
| `index` | recommended | Integer position within the doc (drives `NEXT` edges) |
| `source_path` | optional | Original file path (stored as metadata) |
| `source_kind` | optional | File type, e.g. `pdf` (stored as metadata) |
| `n_chunks` | optional | Total chunks in the document (stored as metadata) |

## Layout

```
main.py                     # entry point (CLI)
chunks_data/                # pre-built .jsonl chunks
src/
  config.py                 # pydantic configuration models
  schema.py                 # Chunk / ProcessedDocument
  factory/                  # LLM + embeddings factories
  ingestion/                # chunks loader, embedder, graph miner
  agents/graph_extractor.py # LLM agent that extracts the graph
  prompts/graph_extractor.py# extraction prompt (ontology-aware)
  graph/                    # graph model, Neo4j store, graph-DS metrics
  utils/logger.py
```
