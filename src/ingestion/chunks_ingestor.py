import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.schema import Chunk, ProcessedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunksIngestor:
    """
    Loads a pre-built JSONL chunks dataset directly into ProcessedDocument objects,
    bypassing document loading, cleaning, and chunking steps.

    Expected JSONL fields per record:
      - text         (required)  chunk text
      - doc_id       (required)  used as ProcessedDocument.filename
      - chunk_id     (required)  original chunk identifier (stored in metadata)
      - index        (optional)  integer position within doc — used as Chunk.chunk_id
                                 so that NEXT relationships work in the graph
      - source_path  (optional)  stored in document metadata
      - source_kind  (optional)  stored in document metadata
      - n_chunks     (optional)  stored in document metadata
    """

    def _parse_lines(self, lines: Iterable[str]) -> List[ProcessedDocument]:
        docs_map: Dict[str, ProcessedDocument] = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            text = record["text"]
            doc_id = record.get("doc_id") or Path(record.get("source_path", "unknown")).stem

            # prefer integer index so NEXT graph relationships resolve correctly
            chunk_id = record.get("index", record.get("chunk_id"))

            chunk = Chunk(
                chunk_id=chunk_id,
                text=text,
                filename=doc_id,
            )

            if doc_id not in docs_map:
                metadata = {k: record.get(k) for k in ("source_path", "source_kind", "n_chunks") if record.get(k) is not None}
                docs_map[doc_id] = ProcessedDocument(
                    filename=doc_id,
                    source="",
                    metadata=metadata,
                    chunks=[],
                )

            docs_map[doc_id].chunks.append(chunk)

        total_chunks = sum(len(d.chunks) for d in docs_map.values())
        logger.info(f"Loaded {total_chunks} chunks across {len(docs_map)} documents.")
        return list(docs_map.values())

    def load_from_file(self, filepath: str) -> List[ProcessedDocument]:
        """Load chunks from a single .jsonl file on disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            return self._parse_lines(f)

    def load_from_folder(self, folder: str) -> List[ProcessedDocument]:
        """Load chunks from all .jsonl files inside a folder."""
        from pathlib import Path
        all_docs: List[ProcessedDocument] = []
        paths = sorted(Path(folder).glob("*.jsonl"))
        if not paths:
            logger.warning(f"No .jsonl files found in {folder}")
            return all_docs
        for path in paths:
            logger.info(f"Loading {path.name}...")
            all_docs.extend(self.load_from_file(str(path)))
        logger.info(f"Total: {sum(len(d.chunks) for d in all_docs)} chunks across {len(all_docs)} documents from folder.")
        return all_docs

    def load_from_bytes(self, content: bytes) -> List[ProcessedDocument]:
        """Load chunks from raw bytes (e.g. Streamlit file_uploader buffer)."""
        return self._parse_lines(content.decode("utf-8").splitlines())
