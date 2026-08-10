"""
Which documents are already in the Knowledge Graph.

Extraction costs hours per document, and `_create_document_node` uses CREATE,
not MERGE — so a second `python main.py` over the same folder does not just
waste that time, it duplicates the document in the graph. This ledger is what
makes the build re-runnable: load everything, skip what is already in, extract
only the rest.

    python -m src.ingestion.build_ledger              # what has been built
    python -m src.ingestion.build_ledger --forget ID  # force one rebuild

A document is keyed by `doc_id` + a digest of its chunks, NOT by file name or
file bytes. Re-uploading the same content under a new name is not a rebuild;
revising the text is. The ledger is also reconciled against Neo4j on every run
(see `reconcile`), so wiping the graph does not leave it claiming a document is
there when it is not.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from src.schema import ProcessedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LEDGER_PATH = "build_ledger.json"
LEDGER_VERSION = 1


def content_digest(doc: ProcessedDocument) -> str:
    """
    Fingerprint of a document's chunk ids and texts, order-independent so a
    reshuffled file is not mistaken for a new one.

    Deliberately NOT including `extraction_eligible`, even though it decides
    which chunks reach the LLM: mixing it in changes the digest of every
    already-built document (the paper corpus carries the flag too), so a run
    that only re-tuned the threshold would re-extract them from scratch — and
    `--rebuild` CREATEs Document nodes rather than MERGEing them. The cost of
    that outweighs catching a threshold change automatically; re-tune with
    `--rebuild` against a wiped graph when you need it.
    """
    digest = hashlib.sha256()
    for chunk in sorted(doc.chunks or [], key=lambda c: str(c.chunk_id)):
        digest.update(str(chunk.chunk_id).encode("utf-8"))
        digest.update(b"\x00")
        digest.update((chunk.text or "").encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def doc_type_of(doc: ProcessedDocument) -> str:
    """The folder a document was loaded from — "paper", "meeting", or "" ."""
    return Path(doc.source).parent.name if doc.source else ""


@dataclass
class LedgerEntry:
    doc_id: str
    digest: str
    doc_type: str = ""
    source_file: str = ""
    n_chunks: int = 0
    built_at: str = ""


class BuildLedger:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or os.getenv("BUILD_LEDGER_PATH", DEFAULT_LEDGER_PATH))
        self.entries: Dict[str, LedgerEntry] = {}
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for doc_id, fields in (raw.get("documents") or {}).items():
                self.entries[doc_id] = LedgerEntry(doc_id=doc_id, **fields)
        except Exception as e:
            # A corrupt ledger must not block a build: the worst case of
            # starting empty is redoing work, while refusing to start means no
            # build at all. Keep the old file so it can be inspected.
            logger.warning(f"Could not read build ledger {self.path} ({e}) — treating every document as unbuilt.")
            self.entries = {}

    def save(self) -> None:
        payload = {
            "version": LEDGER_VERSION,
            "documents": {
                doc_id: {k: v for k, v in asdict(entry).items() if k != "doc_id"}
                for doc_id, entry in sorted(self.entries.items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.part")
        try:
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    # -- queries -------------------------------------------------------------

    def is_built(self, doc: ProcessedDocument, digest: Optional[str] = None) -> bool:
        entry = self.entries.get(doc.filename)
        return entry is not None and entry.digest == (digest or content_digest(doc))

    def entry(self, doc_id: str) -> Optional[LedgerEntry]:
        return self.entries.get(doc_id)

    # -- mutations -----------------------------------------------------------

    def record(self, doc: ProcessedDocument, digest: Optional[str] = None) -> LedgerEntry:
        entry = LedgerEntry(
            doc_id=doc.filename,
            digest=digest or content_digest(doc),
            doc_type=doc_type_of(doc),
            source_file=doc.source or "",
            n_chunks=len(doc.chunks or []),
            built_at=datetime.now(timezone.utc).isoformat(),
        )
        self.entries[entry.doc_id] = entry
        return entry

    def forget(self, doc_id: str) -> bool:
        return self.entries.pop(doc_id, None) is not None

    def reconcile(self, filenames_in_graph: Iterable[str]) -> List[str]:
        """
        Drop entries for documents the graph no longer holds, returning their
        ids. Without this, `MATCH (n) DETACH DELETE n` would leave a ledger that
        skips every document — a wiped graph that can never be refilled.
        """
        present = set(filenames_in_graph)
        stale = [doc_id for doc_id in self.entries if doc_id not in present]
        for doc_id in stale:
            del self.entries[doc_id]
        return stale


# -- CLI ---------------------------------------------------------------------

def _print(ledger: BuildLedger) -> None:
    if not ledger.entries:
        print(f"{ledger.path}: nothing built yet.")
        return
    print(f"{ledger.path}: {len(ledger.entries)} document(s) built\n")
    width = max(len(doc_id) for doc_id in ledger.entries)
    for entry in sorted(ledger.entries.values(), key=lambda e: e.built_at):
        built = entry.built_at[:19].replace("T", " ")
        print(f"  {entry.doc_id:<{width}}  {entry.doc_type or '—':<8} {entry.n_chunks:>5} chunks  {built}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or edit the Knowledge Graph build ledger.")
    parser.add_argument("--path", default=None, help=f"Ledger file (default: BUILD_LEDGER_PATH or ./{DEFAULT_LEDGER_PATH}).")
    parser.add_argument("--forget", metavar="DOC_ID", action="append", default=[],
                        help="Drop a document so the next build re-extracts it. Repeatable.")
    parser.add_argument("--clear", action="store_true", help="Drop every entry.")
    args = parser.parse_args()

    ledger = BuildLedger(path=args.path)

    if args.clear:
        n = len(ledger.entries)
        ledger.entries.clear()
        ledger.save()
        print(f"Cleared {n} entr{'y' if n == 1 else 'ies'}.")
        return

    if args.forget:
        for doc_id in args.forget:
            print(f"{'forgot' if ledger.forget(doc_id) else 'not in ledger'}: {doc_id}")
        ledger.save()
        return

    _print(ledger)


if __name__ == "__main__":
    main()
