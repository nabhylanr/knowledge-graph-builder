# Chunk `.jsonl` Schema — the producer/pipeline contract

This is what a chunk producer (Maruf's `academic-pdf-chunker`, Linus' chunker, …)
must emit for `knowledge-graph-builder` to ingest it correctly.

**Source of truth is the code**, not this page:
[`src/ingestion/chunk_record.py`](../src/ingestion/chunk_record.py). If the two
disagree, the code wins.

Check any file against it before sending:

```bash
python -m src.ingestion.validate path/to/file.jsonl
```

Exit code is `0` if clean, `1` if the file has errors. It only needs `pydantic`
installed — you do not need the rest of the pipeline to run it.

---

## Format

One JSON **object per line** (JSONL), UTF-8, no trailing commas, no wrapping
array. One file may hold several documents (grouped by `doc_id`), but one file
per document is the convention here.

## Fields

| Field | Status | Type | What it does |
|---|---|---|---|
| `text` | **required** | string | The chunk body. This is what gets embedded and sent to the extraction LLM. |
| `doc_id` | **required in practice** | string | Grouping key. Becomes `Document.filename` in Neo4j and ties every chunk of one document together. Falls back to `Path(source_path).stem` if absent — do not rely on that. |
| `index` | **strongly recommended** | integer | 0-based position in the document. See the warning below. |
| `n_chunks` | recommended | integer | Total chunks in the document. Used as a self-check: the validator flags a mismatch against the real record count. |
| `source_path` | recommended | string | Original file path. Stored on the `Document` node. |
| `source_kind` | recommended | string | e.g. `"pdf"`. Sets the `format` property of the `Source` node. Missing → every Source is labelled `"unknown"`. |
| `chunk_id` | optional | string/int | The producer's own id (e.g. `"doc::0000"`). Only used as a fallback when `index` is missing. |
| `page` | optional | integer | Carried, not consumed yet. |
| `section` | optional | string | Carried, not consumed yet. |
| `content_role` | optional | string | e.g. `"abstract"`, `"references"`. Carried, not consumed yet. |
| `extraction_eligible` | optional | boolean (default `true`) | **The pipeline now acts on this.** `false` = store and embed the chunk, but never send it to the extraction LLM. See the section below — its meaning is reconciled at ingest, not taken at face value. |
| `date` | doc-level | string | ISO 8601 (`"2026-01-16"`). Lands on the `Document` node and is promoted to `Source.year`, which the supersession pass requires. Read from the first record of each `doc_id`. |
| `series` | doc-level | string | Recurring-meeting name (`"llm-bi-weekly-meeting"`). Lands on the `Document` node; `PRECEDES` chains documents sharing a series in `date` order. |
| `quality_notes` | optional | string[] | Carried, not consumed yet. |

### Accepted spellings

Both producers' names resolve to the same field:

| Canonical | Also accepted |
|---|---|
| `source_path` | `source_file` |
| `source_kind` | `source_type` |

### `extraction_eligible` — the pipeline's meaning vs yours

The two producers using this field mean different things by it, and only one
matches what the pipeline does with it (skip the LLM call):

| Producer | Sets `false` for | Honoured? |
|---|---|---|
| meeting adapter | chunks with under 70 characters of speech — a transcript's `"you"`, `"Thank you."`. Nothing to extract. | yes |
| `academic-pdf-chunker` | **any** chunk carrying a `quality_notes` entry, including cosmetic PDF damage (`pdf_ligature`, `line_break_hyphenation`) | only partly |

Most of the second kind are readable paragraphs whose text merely renders as
`"Thisfinding"` instead of `"This finding"`. Taking those at face value drops 625
chunks of real content and nearly erases five scanned papers, so
`ChunkRecord.effective_extraction_eligible` overrides `false` back to eligible
**when it is fully explained by cosmetic notes**. A `false` with no notes behind
it — `table_artifact`, `publisher_boilerplate`, `erratum`, and everything the
meeting adapter marks — is always honoured.

If you are a producer: set `false` only when a chunk has **nothing to extract**.
Put rendering defects in `quality_notes` and leave the chunk eligible.

### Doc-level fields

`date` and `series` describe the document, not the chunk, but this schema is
per-record — so repeat them on every record. The ingestor reads them off the
first record of each `doc_id` (same as `source_path` / `source_kind` /
`n_chunks`) and writes them onto the `Document` node.

They exist because the extraction prompt is forbidden to guess a date that is
not in the text, and a meeting's date is essentially never spoken aloud. Without
them arriving as metadata the graph has no time axis: no `PRECEDES` chain across
a recurring meeting, and no `Source.year`, which means the supersession pass
skips every pair involving that document.

### Unknown fields

Extra fields are **allowed and preserved on the record**, but the pipeline does
not read them — they never reach Neo4j. The validator lists them so nothing is
dropped silently. (Linus' `char_start`, `char_end`, `sentence_indices` land here.)

---

## Why `index` matters more than it looks

Chunk ordering in the graph is built by
[`_create_next_relationships`](../src/graph/knowledge_graph.py#L260), which matches
`chunk_id: c1.chunk_id + 1` — **integer arithmetic in Cypher**.

So:

- `index: 0, 1, 2, …` → `NEXT` chain is built, "what came before this passage" works.
- only a string `chunk_id` like `"doc::0000"` → the match never resolves. No error,
  no warning at runtime — you get a graph with **zero** `NEXT` relationships and
  nothing tells you.

Gaps have the same effect locally: an `index` jumping 4 → 6 breaks the chain at
that point. The validator reports both cases.

---

## What counts as an error vs a warning

**Errors** (file is rejected — fix before sending):

- a line that is not valid JSON, or not a JSON object
- `text` missing
- two records with the same `doc_id` + effective chunk id — the second silently
  overwrites the first in the graph
- neither `doc_id` nor `source_path` present, so chunks cannot be grouped

**Warnings** (file ingests, but the graph is degraded):

- missing / gapped / non-zero-based `index` → broken `NEXT` chain
- empty `text`
- `n_chunks` disagreeing with the actual record count
- no `source_kind` → `Source` nodes labelled `"unknown"`

---

## Example (one line, wrapped here for readability)

```json
{
  "text": "The Internet of Things (IoT) became the most impactful development worldwide...",
  "doc_id": "Thesis_M10801863_Edwin_7e1ab30c",
  "index": 0,
  "n_chunks": 168,
  "source_file": "C:\\academic-pdf-chunker\\data\\raw\\Thesis_M10801863_Edwin.pdf",
  "source_type": "pdf",
  "page": 1,
  "section": null,
  "content_role": "abstract",
  "extraction_eligible": true,
  "quality_notes": []
}
```

---

## Where files go

Locally: `chunks_data/<producer>/<doc_id>.jsonl` — e.g. `chunks_data/maruf/`.

To hand a file over, see [chunk_sync.md](./chunk_sync.md) — upload to Supabase
Storage and record it in the manifest table for its kind (`paper_chunk_uploads`
or `meeting_chunk_uploads`); the receiving side pulls, verifies the checksum and
re-runs this validator before the file is accepted.
