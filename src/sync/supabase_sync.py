"""
Pulls chunk files a producer uploaded to Supabase down onto this machine.

Files land in `chunks_data/<doc_type>/`, one folder per kind of document
(`paper`, `meeting`) rather than one per producer — see `resolve_doc_type` for
how a row is sorted into one.

Polling, not webhooks: the state lives in the `chunk_uploads` row, so a laptop
that was asleep for three days still picks up everything it missed on the next
run. A delivered-once event would simply have been lost.

Deliberately stops at `downloaded`. It never starts a KG build — a build is a
multi-hour LLM job, and two uploads arriving close together would otherwise
launch two of them against the same Ollama and Neo4j. Deciding when to build
stays manual.

See docs/chunk_sync.md for setup and db/supabase_schema.sql for the table.
"""

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.ingestion.validate import MAX_LISTED, validate_lines
from src.utils.logger import get_logger

logger = get_logger(__name__)

PENDING = "pending"
DOWNLOADED = "downloaded"
BUILT = "built"
FAILED = "failed"

# The local layout under `dest_root`: one folder per kind of document. The
# producer's own folder names are NOT mirrored — where a file came from is the
# producer's business, what it *is* decides where it lands here.
DOC_TYPES = ("paper", "meeting")

# Fallback signals for `resolve_doc_type`, matched as substrings of the
# producer's `source_kind`/`source_type`.
_MEETING_KINDS = ("meeting", "transcript", "minutes", "vtt", "srt", "recording", "audio")
_PAPER_KINDS = ("pdf", "paper", "thesis", "article", "journal", "docx", "tex")


@dataclass
class SyncConfig:
    url: str
    service_key: str
    bucket: str = "chunks"
    dest_root: str = "chunks_data"
    table: str = "chunk_uploads"
    # Used only when nothing else identifies the document — see resolve_doc_type.
    default_doc_type: str = "paper"


@dataclass
class SyncOutcome:
    doc_id: str
    status: str
    detail: str = ""
    doc_type: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_file_name(storage_path: str) -> str:
    """
    The name the file gets on disk, refusing anything that would escape the
    destination folder. `storage_path` comes from a row a producer wrote, so it
    is untrusted input — "../../.ssh/authorized_keys" is a valid string.

    Only the base name survives. The local tree is organised by doc_type, not by
    whichever folder the producer happened to upload into.
    """
    rel = PurePosixPath(storage_path)
    if rel.is_absolute() or any(part in ("..", "") for part in rel.parts):
        raise ValueError(f"unsafe storage_path: {storage_path!r}")
    if not rel.name or rel.name in (".", ".."):
        raise ValueError(f"unsafe storage_path: {storage_path!r}")
    return rel.name


def resolve_doc_type(
    row: Dict[str, Any],
    storage_path: str,
    source_kinds: Iterable[Optional[str]],
    default: str = "paper",
) -> Tuple[str, str]:
    """
    Decide whether a file belongs in `paper/` or `meeting/`. Returns
    `(doc_type, why)`; `why` is logged so a misfiled document can be traced back
    to the signal that put it there.

    Four signals, most authoritative first — the layering is what lets this work
    before every producer has caught up:

    1. `doc_type` on the manifest row. Explicit, and the only one that cannot be
       wrong; producers should fill it (see db/supabase_schema.sql).
    2. A `paper/` or `meeting/` prefix on the bucket path. Costs the producer
       nothing but an upload path.
    3. The `source_kind`/`source_type` the chunk file already carries — a `pdf`
       is a paper, a `transcript` is a meeting.
    4. `default`. Never blocks a download; logged loudly because a wrong guess
       here is silent otherwise.
    """
    declared = str(row.get("doc_type") or "").strip().lower()
    if declared in DOC_TYPES:
        return declared, "row.doc_type"

    parts = PurePosixPath(storage_path).parts
    if len(parts) > 1:
        prefix = parts[0].strip().lower().rstrip("s")  # accepts "papers/" too
        if prefix in DOC_TYPES:
            return prefix, f"storage_path prefix {parts[0]!r}"

    for kind in source_kinds:
        normalized = str(kind or "").strip().lower()
        if not normalized:
            continue
        if any(marker in normalized for marker in _MEETING_KINDS):
            return "meeting", f"source_kind={kind!r}"
        if any(marker in normalized for marker in _PAPER_KINDS):
            return "paper", f"source_kind={kind!r}"

    return default, "fallback default — no doc_type, no path prefix, no usable source_kind"


def _write_atomically(dest: Path, content: bytes) -> None:
    """
    Write via a temp file + rename so an interrupted sync never leaves a
    half-written .jsonl that looks complete to the next build.

    The temp name carries the pid because the listener daemon and the scheduled
    poll can both be draining the queue at the same moment. Downloading the same
    row twice is harmless (identical bytes, idempotent status update), but a
    shared temp path would let one writer rename the other's half-written file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    try:
        tmp.write_bytes(content)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


class ChunkSync:
    def __init__(self, conf: SyncConfig):
        from supabase import create_client  # imported here so the rest of the pipeline needs no supabase install

        self.conf = conf
        self.client = create_client(conf.url, conf.service_key)
        logger.info(f"Connected to Supabase project at {conf.url}")

    # -- queries ------------------------------------------------------------

    def fetch_pending(self, limit: int = 0) -> List[Dict[str, Any]]:
        query = (
            self.client.table(self.conf.table)
            .select("*")
            .eq("status", PENDING)
            .order("uploaded_at", desc=False)
        )
        if limit and limit > 0:
            query = query.limit(limit)
        return query.execute().data or []

    def _mark(self, row_id: int, status: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {"status": status, **fields}
        self.client.table(self.conf.table).update(payload).eq("id", row_id).execute()

    def mark_built(self, doc_id: str) -> int:
        """Close the loop after a KG build. Returns how many rows moved."""
        result = (
            self.client.table(self.conf.table)
            .update({"status": BUILT, "built_at": _now()})
            .eq("doc_id", doc_id)
            .eq("status", DOWNLOADED)
            .execute()
        )
        return len(result.data or [])

    # -- the work -----------------------------------------------------------

    def sync_one(self, row: Dict[str, Any], dry_run: bool = False) -> SyncOutcome:
        doc_id = row.get("doc_id", "?")
        storage_path = row["storage_path"]

        try:
            file_name = _safe_file_name(storage_path)
            content = self.client.storage.from_(self.conf.bucket).download(storage_path)
        except Exception as e:
            detail = f"download failed: {e}"
            if not dry_run:
                self._mark(row["id"], FAILED, error=detail)
            return SyncOutcome(doc_id, FAILED, detail)

        digest = hashlib.sha256(content).hexdigest()
        if digest != row["sha256"]:
            detail = f"sha256 mismatch — expected {row['sha256'][:12]}…, got {digest[:12]}…"
            if not dry_run:
                self._mark(row["id"], FAILED, error=detail)
            return SyncOutcome(doc_id, FAILED, detail)

        try:
            report = validate_lines(content.decode("utf-8").splitlines(), path=storage_path)
        except UnicodeDecodeError as e:
            detail = f"file is not valid UTF-8: {e}"
            if not dry_run:
                self._mark(row["id"], FAILED, error=detail)
            return SyncOutcome(doc_id, FAILED, detail)

        if not report.ok:
            # The producer can read `error` on their own row, so put the actual
            # failing lines there rather than making them ask what went wrong.
            detail = f"{len(report.errors)} schema error(s): " + " | ".join(report.errors[:MAX_LISTED])
            if not dry_run:
                self._mark(row["id"], FAILED, error=detail[:4000])
            return SyncOutcome(doc_id, FAILED, detail)

        for warning in report.warnings:
            logger.warning(f"{doc_id}: {warning}")

        doc_type, why = resolve_doc_type(
            row,
            storage_path,
            [doc.source_kind for doc in report.docs.values()],
            default=self.conf.default_doc_type,
        )
        dest = Path(self.conf.dest_root) / doc_type / file_name
        log = logger.info if why != "row.doc_type" else logger.debug
        log(f"{doc_id}: filed as {doc_type} ({why})")

        if dry_run:
            return SyncOutcome(doc_id, DOWNLOADED, f"would write {dest} ({report.n_records} records)", doc_type)

        _write_atomically(dest, content)
        # Only echo the resolved type back when the column exists — the update
        # would otherwise fail wholesale on a database that has not run the
        # doc_type migration yet, taking the status change down with it.
        echo = {"doc_type": doc_type} if "doc_type" in row else {}
        self._mark(row["id"], DOWNLOADED, downloaded_at=_now(), error=None, **echo)
        return SyncOutcome(doc_id, DOWNLOADED, f"{dest} ({report.n_records} records)", doc_type)

    def run(self, limit: int = 0, dry_run: bool = False) -> List[SyncOutcome]:
        rows = self.fetch_pending(limit=limit)
        if not rows:
            logger.info("Nothing pending.")
            return []

        logger.info(f"{len(rows)} pending upload(s).")
        outcomes = []
        for row in rows:
            outcome = self.sync_one(row, dry_run=dry_run)
            if outcome.status == DOWNLOADED:
                logger.info(f"{outcome.doc_id}: {outcome.detail}")
            else:
                logger.error(f"{outcome.doc_id}: {outcome.detail}")
            outcomes.append(outcome)
        return outcomes


def build_sync_config(dest_root: Optional[str] = None) -> SyncConfig:
    """Reads the SUPABASE_* settings from the environment (.env)."""
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not service_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set — see .env_example and docs/chunk_sync.md"
        )
    default_doc_type = os.getenv("DEFAULT_DOC_TYPE", "paper").strip().lower()
    if default_doc_type not in DOC_TYPES:
        raise RuntimeError(f"DEFAULT_DOC_TYPE must be one of {DOC_TYPES}, got {default_doc_type!r}")

    return SyncConfig(
        url=url,
        service_key=service_key,
        bucket=os.getenv("SUPABASE_BUCKET", "chunks"),
        dest_root=dest_root or os.getenv("CHUNKS_DEST_DIR", "chunks_data"),
        default_doc_type=default_doc_type,
    )
