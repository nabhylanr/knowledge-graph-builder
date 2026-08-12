"""
Which chunks of an in-progress document are already in the Knowledge Graph.

Distinct from `BuildLedger` (src/ingestion/build_ledger.py), and the two are not
interchangeable: the ledger marks a document "fully built", and its entry is
written once, at the end. This checkpoint is written incrementally, DURING a
document's processing, so a crash at chunk 80 of 129 costs the in-flight chunk
rather than the 80 already extracted. Once a document is fully built its
checkpoint entry is redundant with the ledger and is removed — otherwise this
file grows without bound across a long corpus.

    python -m src.ingestion.chunk_checkpoint          # what is half-built
    python -m src.ingestion.chunk_checkpoint --forget ID

"Done" here means WRITTEN TO NEO4J, not merely extracted — see
`GraphMiner.mine_and_store_doc_chunks` for why the mark has to come after the
write and never before it.

An entry also carries the sanitizer state that spans a document's chunks
(`has_source_state`, `topic_registry`, `source_meta_state` — see
`sanitize_graph`). Without that, resuming would restart those counters from
zero and a resumed document could end up with more than MAX_HAS_SOURCE
has_source edges, or re-canonicalize a Topic under a second spelling. The chunk
list alone is not enough to resume faithfully.
"""

import argparse
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHECKPOINT_PATH = "chunk_checkpoint.json"
CHECKPOINT_VERSION = 1

ChunkId = Union[int, str]


def _doc_key(filename: str, document_version: int) -> str:
    return f"{filename}|{document_version}"


class ChunkCheckpoint:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or os.getenv("CHUNK_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH))
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = raw.get("documents") or {}
        except Exception as e:
            # Same reasoning as BuildLedger._load: a corrupt checkpoint must not
            # block a build. The worst case of starting empty is re-extracting a
            # document that was half done, which is exactly what happened before
            # this file existed.
            logger.warning(f"Could not read chunk checkpoint {self.path} ({e}) — starting from no progress.")
            self.entries = {}

    def _save(self) -> None:
        """Caller must hold `self._lock`."""
        payload = {
            "version": CHECKPOINT_VERSION,
            "documents": {key: self.entries[key] for key in sorted(self.entries)},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.part")
        try:
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)  # atomic — a crash mid-write must not corrupt the file that exists to survive crashes
        finally:
            Path(tmp).unlink(missing_ok=True)

    # -- queries -------------------------------------------------------------

    def _entry(self, filename: str, document_version: int, digest: str = "") -> Dict[str, Any]:
        """
        The stored entry, or an empty one when there is none — or when the
        document's content has changed under it.

        The digest check is what makes a checkpoint id safe to trust: chunk ids
        are positions within a document, so if the source .jsonl is revised
        without `document_version` being bumped (nothing bumps it automatically —
        it is a manual field on ProcessedDocument), chunk 40 of the new content
        is a different chunk from the checkpointed chunk 40. Rather than resume
        onto the wrong text, a digest mismatch discards the entry and the
        document is re-extracted from scratch.
        """
        entry = self.entries.get(_doc_key(filename, document_version))
        if not entry:
            return {}
        stored = entry.get("digest", "")
        if digest and stored and stored != digest:
            logger.warning(
                f"Chunk checkpoint for {filename} was written for different content "
                f"(digest {stored[:12]}… != {digest[:12]}…) — ignoring it and rebuilding the document."
            )
            return {}
        return entry

    def done_chunk_ids(self, filename: str, document_version: int, digest: str = "") -> Set[ChunkId]:
        """Chunk ids of this document already written to Neo4j."""
        return set(self._entry(filename, document_version, digest).get("chunks", []))

    def sanitizer_state(self, filename: str, document_version: int, digest: str = "") -> Dict[str, dict]:
        """
        The cross-chunk `sanitize_graph` state as of the last checkpointed chunk:
        `{"has_source": ..., "topics": ..., "source_meta": ...}`, or empty dicts
        when there is nothing to resume from. See this module's docstring.
        """
        state = self._entry(filename, document_version, digest).get("state") or {}
        return {
            "has_source": dict(state.get("has_source") or {}),
            "topics": dict(state.get("topics") or {}),
            "source_meta": {k: dict(v) for k, v in (state.get("source_meta") or {}).items()},
        }

    def documents(self) -> List[str]:
        return sorted(self.entries)

    # -- mutations -----------------------------------------------------------

    def mark_done(
        self,
        filename: str,
        document_version: int,
        chunk_id: ChunkId,
        digest: str = "",
        state: Optional[Dict[str, dict]] = None,
    ) -> None:
        """
        Record one chunk as written and persist immediately.

        Persisting on every call is the whole point — surviving a crash on the
        very next line is what this file is for, and an in-memory-only update
        would defeat it. `state` is the caller's live sanitizer state, snapshotted
        alongside so a resume continues the has_source cap and Topic registry
        rather than restarting them.
        """
        with self._lock:
            key = _doc_key(filename, document_version)
            entry = self.entries.get(key)
            if entry is None or (digest and entry.get("digest") and entry["digest"] != digest):
                entry = {"digest": digest, "chunks": [], "state": {}}
                self.entries[key] = entry
            entry["digest"] = digest or entry.get("digest", "")
            # Sorted by str() because chunk ids may be ints or strings
            # (Chunk.chunk_id is Union[int, str]) and a mixed list is unsortable.
            entry["chunks"] = sorted(set(entry.get("chunks", [])) | {chunk_id}, key=str)
            if state is not None:
                entry["state"] = state
            self._save()

    def clear_document(self, filename: str, document_version: int) -> bool:
        """
        Drop a document's progress. Called once it is fully built and recorded in
        the BuildLedger (the entry is redundant from then on), and by `--rebuild`
        before it starts, so a rebuild really does redo every chunk.
        """
        with self._lock:
            removed = self.entries.pop(_doc_key(filename, document_version), None) is not None
            if removed:
                self._save()
            return removed


# -- CLI ---------------------------------------------------------------------

def _print(checkpoint: ChunkCheckpoint) -> None:
    if not checkpoint.entries:
        print(f"{checkpoint.path}: no document is half-built.")
        return
    print(f"{checkpoint.path}: {len(checkpoint.entries)} document(s) in progress\n")
    width = max(len(key) for key in checkpoint.entries)
    for key in sorted(checkpoint.entries):
        n = len(checkpoint.entries[key].get("chunks", []))
        print(f"  {key:<{width}}  {n:>5} chunk(s) written")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or edit the chunk-level build checkpoint.")
    parser.add_argument("--path", default=None,
                        help=f"Checkpoint file (default: CHUNK_CHECKPOINT_PATH or ./{DEFAULT_CHECKPOINT_PATH}).")
    parser.add_argument("--forget", metavar="DOC_KEY", action="append", default=[],
                        help="Drop a document's progress so the next build re-extracts every chunk. "
                             "DOC_KEY is 'filename|document_version' as listed. Repeatable.")
    parser.add_argument("--clear", action="store_true", help="Drop every entry.")
    args = parser.parse_args()

    checkpoint = ChunkCheckpoint(path=args.path)

    if args.clear:
        n = len(checkpoint.entries)
        with checkpoint._lock:
            checkpoint.entries.clear()
            checkpoint._save()
        print(f"Cleared {n} entr{'y' if n == 1 else 'ies'}.")
        return

    if args.forget:
        for key in args.forget:
            with checkpoint._lock:
                removed = checkpoint.entries.pop(key, None) is not None
                if removed:
                    checkpoint._save()
            print(f"{'forgot' if removed else 'not in checkpoint'}: {key}")
        return

    _print(checkpoint)


if __name__ == "__main__":
    main()
