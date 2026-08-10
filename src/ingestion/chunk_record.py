"""
The wire format for a chunks `.jsonl` file — one `ChunkRecord` per line.

This is the contract between a chunk producer (Maruf's academic-pdf-chunker,
Linus' chunker, ...) and this pipeline. Producers emit slightly different field
names for the same thing, so the aliases below are the single place where those
spellings are reconciled; everything downstream sees one shape.

Deliberately depends on nothing but pydantic — `to_chunk()` imports `src.schema`
lazily — so a producer can `pip install pydantic` and run
`python -m src.ingestion.validate` on their own machine without pulling in
langchain, neo4j and the rest of the pipeline.

See docs/chunk_schema.md for the human-readable version of this contract.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# Doc-level metadata copied onto the Document node in Neo4j. Taken from the first
# record of each doc_id — these are expected to be constant across a document.
#
# `_create_document_node` writes these with `SET d += $metadata`, so anything
# added here lands on the Document node verbatim. `date` and `series` are what
# `_create_precedes_relationships` chains a recurring meeting by, and `date` is
# also promoted to `Source.year`, which the supersession pass requires.
DOC_METADATA_FIELDS = ("source_path", "source_kind", "n_chunks", "date", "series")

# `quality_notes` values that describe how the text was RENDERED, not whether it
# says anything. See `effective_extraction_eligible` for why this list exists.
#
# Deliberately excludes `layout_fragment`: the one sampled is a bibliography line
# ("Chan, K. W., Yim, C. K., & Lam, S."), i.e. genuinely not worth extracting, so
# those 32 chunks keep the producer's verdict. Move it in here if a closer look
# says otherwise — that is the only edit this decision should ever need.
COSMETIC_QUALITY_NOTES = frozenset({"pdf_ligature", "line_break_hyphenation"})


class ChunkRecord(BaseModel):
    """
    One chunk as it arrives on disk.

    Only `text` is strictly required; `doc_id` and `index` are required in
    practice (see the warnings raised by `src.ingestion.validate`) because
    without them chunks cannot be grouped into documents or chained with NEXT.

    `extra="allow"` is intentional: an unrecognised field is a producer telling
    us something we do not consume yet, which is a note to the reader, not an
    error. The validator lists them so nothing is silently dropped.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    text: str

    doc_id: Optional[str] = None
    chunk_id: Optional[Union[int, str]] = None
    index: Optional[int] = None
    n_chunks: Optional[int] = None

    # `source_file`/`source_type` is Maruf's spelling, `source_path`/`source_kind`
    # is Linus'. Both land on the same field — before these aliases existed
    # Maruf's values were dropped, which left every Source node's format "unknown".
    source_path: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("source_path", "source_file"),
    )
    source_kind: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("source_kind", "source_type"),
    )

    # Doc-level facts the producer knows and the text does not state. A meeting's
    # date is never spoken aloud and the extraction prompt is forbidden to guess
    # one, so without these arriving as metadata the graph has no time axis at
    # all. Repeated on every record (this schema is per-chunk); the ingestor
    # reads them off the first record of each doc_id. See DOC_METADATA_FIELDS.
    date: Optional[str] = None      # ISO 8601, e.g. "2026-01-16"
    series: Optional[str] = None    # recurring-meeting name, e.g. "llm-bi-weekly-meeting"

    # Producer-side annotations. Carried through so they are visible in the
    # validator report; the extraction pipeline does not act on them yet.
    page: Optional[int] = None
    section: Optional[str] = None
    content_role: Optional[str] = None
    quality_notes: List[str] = Field(default_factory=list)

    # The one producer annotation the pipeline DOES act on: `false` means store
    # the chunk but skip the extraction LLM call for it (see `Chunk` in
    # src/schema.py and GraphMiner.mine_graph_from_doc_chunks).
    extraction_eligible: bool = True

    @property
    def effective_doc_id(self) -> str:
        """The value used as `ProcessedDocument.filename` — the grouping key."""
        if self.doc_id:
            return self.doc_id
        return Path(self.source_path or "unknown").stem

    @property
    def effective_chunk_id(self) -> Union[int, str]:
        """
        What becomes `Chunk.chunk_id` in the graph.

        `index` wins over `chunk_id` because NEXT relationships are built with
        `chunk_id: c1.chunk_id + 1` (see knowledge_graph._create_next_relationships),
        which only resolves for integers. A string id like "doc::0000" would
        silently produce a document with no NEXT chain at all.
        """
        return self.index if self.index is not None else self.chunk_id

    @property
    def effective_extraction_eligible(self) -> bool:
        """
        Whether the pipeline should actually spend an extraction call on this
        chunk — the producer's flag, reconciled to this pipeline's meaning of it.

        The two producers use `extraction_eligible` for different things, and
        only one of them matches what the pipeline does with it (skip the LLM):

        - The meeting adapter sets it false for chunks with no content at all —
          a transcript's "you" or "Thank you.". Nothing to extract. Honoured.
        - Maruf's academic-pdf-chunker sets it false whenever ANY `quality_notes`
          entry is present (verified: 657/657 of its flagged chunks have notes,
          with no residual — it is a rule, not a human judgement). Most of those
          notes are `pdf_ligature` / `line_break_hyphenation`, i.e. "Thisfinding"
          instead of "This finding". The paragraph is perfectly readable and
          carries the paper's argument; taking the flag at face value would drop
          625 chunks of real content and nearly erase five scanned papers.

        So a false flag is overridden back to eligible only when it is fully
        explained by cosmetic notes. A flag with no notes behind it (the
        producer's `table_artifact` / `publisher_boilerplate` / `erratum` chunks,
        and every chunk the meeting adapter marks) is always honoured.

        Reconciled here rather than by rewriting the .jsonl files because papers
        arrive from Supabase — a re-sync would overwrite any local edit and
        silently bring the problem back.
        """
        if self.extraction_eligible:
            return True
        notes = set(self.quality_notes)
        return bool(notes) and notes <= COSMETIC_QUALITY_NOTES

    @property
    def unknown_fields(self) -> List[str]:
        """Fields present in the record that this schema does not describe."""
        return sorted(self.model_extra or {})

    def doc_metadata(self) -> Dict[str, Any]:
        """Doc-level metadata for the Document node, skipping absent values."""
        return {
            name: value
            for name in DOC_METADATA_FIELDS
            if (value := getattr(self, name)) is not None
        }

    def to_chunk(self):
        """Convert to the pipeline's internal `Chunk`."""
        from src.schema import Chunk  # local import — keeps this module pydantic-only

        return Chunk(
            chunk_id=self.effective_chunk_id,
            text=self.text,
            filename=self.effective_doc_id,
            extraction_eligible=self.effective_extraction_eligible,
        )
