import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.ingestion.chunk_record import ChunkRecord
from src.schema import ProcessedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
            doc_id = record.effective_doc_id

            if doc_id not in docs_map:
                docs_map[doc_id] = ProcessedDocument(
                    filename=doc_id,
                    # Which file on disk this came from. The build ledger records
                    # it, and its parent folder is the document's doc_type.
                    source=source,
                    metadata=record.doc_metadata(),
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
