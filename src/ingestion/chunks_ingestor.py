import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from src.ingestion.chunk_record import ChunkRecord
from src.schema import ProcessedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Narrow, evidence-based fixes only — each rule below was checked against the
# actual chunks_data corpus before being added; a broader "strip anything that
# looks alien" filter was considered and rejected (see chat history) because
# it matched Traditional Chinese thesis title pages (e.g.
# Thesis_M10801107_Yu_Ting_Chiu, 79 CJK characters including an ROC-calendar
# date) as "alien symbols" and would have deleted real content, not noise.
_NULL_BYTE = "\x00"
# JSTOR's semi-transparent download-stamp watermark ("This content downloaded
# from...") extracts with literal null bytes interspersed — verified in
# Franke_1978_Hawthorne_Effect, chunk index 1.
_LIKERT_SCALE_RE = re.compile(r"\b12345\b")
# A Likert-scale response line ("1 2 3 4 5") that PDF extraction merges into
# one token with no separators — verified verbatim in Martinez_2009 and
# Zhao_2015. Exact-match only, deliberately not broadened to other renderings
# ("1-5", spaced digits, ...) without evidence they occur in this corpus —
# broadening speculatively risks eating real numeric content instead.
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_chunk_text(text: str) -> str:
    if not text:
        return text
    text = text.replace(_NULL_BYTE, " ")
    text = _LIKERT_SCALE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


class ChunksIngestor:
    """
    Loads a pre-built JSONL chunks dataset directly into ProcessedDocument objects,
    bypassing document loading, cleaning, and chunking steps.

    The per-record field contract lives in `src.ingestion.chunk_record.ChunkRecord`
    (and in prose in docs/chunk_schema.md). Run
    `python -m src.ingestion.validate <path>` to check a file against it before
    spending hours of extraction on it.
    """

    def _parse_lines(self, lines: Iterable[str], source: str = "") -> List[ProcessedDocument]:
        docs_map: Dict[str, ProcessedDocument] = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            record = ChunkRecord.model_validate(json.loads(line))
            record.text = _clean_chunk_text(record.text)
            doc_id = record.effective_doc_id

            if doc_id not in docs_map:
                metadata = record.doc_metadata()
                # The folder is the document's kind. Recorded on the Document node
                # too, not only in the build ledger — without it Neo4j has no way
                # to tell a paper from a meeting, so "every Decision from a
                # meeting" is not a question the graph can answer.
                doc_type = Path(source).parent.name if source else ""
                if doc_type:
                    metadata["doc_type"] = doc_type

                docs_map[doc_id] = ProcessedDocument(
                    filename=doc_id,
                    # Which file on disk this came from. The build ledger records
                    # it, and its parent folder is the document's doc_type.
                    source=source,
                    metadata=metadata,
                    chunks=[],
                )

            docs_map[doc_id].chunks.append(record.to_chunk())

        total_chunks = sum(len(d.chunks) for d in docs_map.values())
        logger.info(f"Loaded {total_chunks} chunks across {len(docs_map)} documents.")
        return list(docs_map.values())

    def load_from_file(self, filepath: str) -> List[ProcessedDocument]:
        """Load chunks from a single .jsonl file on disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            return self._parse_lines(f, source=str(filepath))

    def load_from_folder(self, folder: str) -> List[ProcessedDocument]:
        """
        Load chunks from every .jsonl file under a folder, recursively.

        Recursive because the dataset is organised in subfolders by document
        kind (`chunks_data/paper/`, `chunks_data/meeting/`) — a flat glob on
        `chunks_data` would quietly find nothing at all.
        """
        all_docs: List[ProcessedDocument] = []
        root = Path(folder)
        paths = sorted(root.rglob("*.jsonl"))
        if not paths:
            logger.warning(f"No .jsonl files found under {folder}")
            return all_docs
        for path in paths:
            logger.info(f"Loading {path.relative_to(root) if path.is_relative_to(root) else path}...")
            all_docs.extend(self.load_from_file(str(path)))
        logger.info(f"Total: {sum(len(d.chunks) for d in all_docs)} chunks across {len(all_docs)} documents from folder.")
        return all_docs

    def load_from_bytes(self, content: bytes) -> List[ProcessedDocument]:
        """Load chunks from raw bytes (e.g. Streamlit file_uploader buffer)."""
        return self._parse_lines(content.decode("utf-8").splitlines())
