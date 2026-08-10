"""
Pulls chunk files a producer uploaded to Supabase down onto this machine.

There is **one manifest table per kind of document** — `paper_chunk_uploads` and
`meeting_chunk_uploads` — and the table a row sits in *is* its doc_type. That is
what decides the local folder: `chunks_data/paper/` or `chunks_data/meeting/`.
Nothing is guessed on this side any more; a producer picks a type by picking a
table, and the two producers have separate queues, so a broken paper upload
cannot hold up meetings.

The two queues carry different file formats, which is the other reason they are
separate. Papers arrive as the pipeline's own `.jsonl`; meetings arrive as the
meeting chunker's `*.chunks.json` and are translated on the way in by
`src.ingestion.meeting_chunks` (the same adapter `scripts/convert_meeting_chunks.py`
uses). Either queue also accepts an already-converted `.jsonl`.

Polling, not webhooks: the state lives in the manifest row, so a laptop that was
asleep for three days still picks up everything it missed on the next run. A
delivered-once event would simply have been lost.

Deliberately stops at `downloaded`. It never starts a KG build — a build is a
multi-hour LLM job, and two uploads arriving close together would otherwise
launch two of them against the same Ollama and Neo4j. Deciding when to build
stays manual.

See docs/chunk_sync.md for setup and db/supabase_schema.sql for the tables.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.ingestion.meeting_chunks import (
    DEFAULT_ELIGIBLE_MIN_CHARS,
    looks_like_meeting_payload,
    records_from_payload,
    to_lines,
)
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

# doc_type -> manifest table. The mapping is the whole routing rule on this side.
DEFAULT_TABLES: Dict[str, str] = {
    "paper": "paper_chunk_uploads",
    "meeting": "meeting_chunk_uploads",
}

# Signals for `infer_doc_type`, matched as substrings of a chunk record's
# `source_kind`/`source_type`.
_MEETING_KINDS = ("meeting", "transcript", "minutes", "vtt", "srt", "recording", "audio")
_PAPER_KINDS = ("pdf", "paper", "thesis", "article", "journal", "docx", "tex")


@dataclass
class SyncConfig:
    url: str
    service_key: str
    bucket: str = "chunks"
    dest_root: str = "chunks_data"
    tables: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TABLES))
    # Meeting chunks with less than this many characters of speech are marked
    # `extraction_eligible: false` during conversion — still stored and embedded,
    # just never sent to the extraction LLM. 0 marks nothing.
    meeting_eligible_min_chars: int = DEFAULT_ELIGIBLE_MIN_CHARS


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


def infer_doc_type(source_kinds: Iterable[Optional[str]]) -> Optional[str]:
    """
    What the chunk records themselves say they are — a `pdf` is a paper, a
    `transcript` is a meeting — or None when nothing in them answers.

    On the receiving side this no longer routes anything (the table does); it is
    only used to notice that a file disagrees with the queue it arrived in.
    """
    for kind in source_kinds:
        normalized = str(kind or "").strip().lower()
        if not normalized:
            continue
        if any(marker in normalized for marker in _MEETING_KINDS):
            return "meeting"
        if any(marker in normalized for marker in _PAPER_KINDS):
            return "paper"
    return None


def resolve_doc_type(
    row: Dict[str, Any],
    storage_path: str,
    source_kinds: Iterable[Optional[str]],
    default: str = "paper",
) -> Tuple[str, str]:
    """
    Producer-side: decide which queue a file belongs in. Returns
    `(doc_type, why)`; `why` is printed so a file that went to the wrong table
    can be traced back to the signal that put it there.

    Three signals, most authoritative first, then a default:

    1. An explicit `doc_type` (the uploader's `--doc-type` flag). Cannot be wrong.
    2. A `paper/` or `meeting/` prefix on the bucket path.
    3. The file's own `source_kind`/`source_type` — see `infer_doc_type`.
    4. `default`. Logged loudly, because a wrong guess here is silent otherwise.
    """
    declared = str(row.get("doc_type") or "").strip().lower()
    if declared in DOC_TYPES:
        return declared, "row.doc_type"

    parts = PurePosixPath(storage_path).parts
    if len(parts) > 1:
        prefix = parts[0].strip().lower().rstrip("s")  # accepts "papers/" too
        if prefix in DOC_TYPES:
            return prefix, f"storage_path prefix {parts[0]!r}"

    inferred = infer_doc_type(source_kinds)
    if inferred:
        return inferred, "the file's own source_kind"

    return default, "fallback default — no doc_type, no path prefix, no usable source_kind"


def prepare_download(
    doc_type: str,
    content: bytes,
    storage_path: str,
    meeting_eligible_min_chars: int = DEFAULT_ELIGIBLE_MIN_CHARS,
) -> Tuple[str, bytes, List[str], List[str]]:
    """
    Turn downloaded bytes into what should land on disk. Returns
    `(file_name, blob, lines, notes)` — `lines` is what gets validated, `blob` is
    what gets written.

    A paper is already in the pipeline's format, so its bytes are written through
    untouched. A meeting is normally the chunker's `*.chunks.json` and is
    converted here; it is written under `<doc_id>.jsonl` rather than the uploaded
    name, because the doc_id is what the rest of the pipeline groups by, and the
    producer's file name is the shared `speaker_transcript`.

    Raises `ValueError` / `UnicodeDecodeError` for anything unreadable; the caller
    turns that into a `failed` row.
    """
    text = content.decode("utf-8")  # UnicodeDecodeError -> caller

    if doc_type == "meeting":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None  # not one JSON document — fall through to .jsonl

        if looks_like_meeting_payload(payload):
            records, notes = records_from_payload(payload, eligible_min_chars=meeting_eligible_min_chars)
            if not records:
                raise ValueError("no chunks left after conversion")
            lines = to_lines(records)
            file_name = f"{records[0]['doc_id']}.jsonl"
            return file_name, ("\n".join(lines) + "\n").encode("utf-8"), lines, notes

    # Everything else is already .jsonl — including a meeting a producer chose to
    # convert on their own side.
    return _safe_file_name(storage_path), content, text.splitlines(), []


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
        logger.info(
            f"Connected to Supabase project at {conf.url} "
            f"(tables: {', '.join(f'{t}={n}' for t, n in sorted(conf.tables.items()))})"
        )

    # -- queries ------------------------------------------------------------

    def fetch_pending(self, limit: int = 0) -> List[Tuple[str, Dict[str, Any]]]:
        """`(doc_type, row)` for everything pending, oldest upload first."""
        pending: List[Tuple[str, Dict[str, Any]]] = []
        for doc_type, table in sorted(self.conf.tables.items()):
            query = (
                self.client.table(table)
                .select("*")
                .eq("status", PENDING)
                .order("uploaded_at", desc=False)
            )
            if limit and limit > 0:
                query = query.limit(limit)
            pending.extend((doc_type, row) for row in (query.execute().data or []))

        # Merged into one queue so `--limit 2` means the two oldest uploads
        # overall, not two per table.
        pending.sort(key=lambda pair: str(pair[1].get("uploaded_at") or ""))
        return pending[:limit] if limit and limit > 0 else pending

    def _mark(self, doc_type: str, row_id: int, status: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {"status": status, **fields}
        self.client.table(self.conf.tables[doc_type]).update(payload).eq("id", row_id).execute()

    def mark_built(self, doc_id: str) -> int:
        """
        Close the loop after a KG build. Returns how many rows moved.

        Both tables are swept: `main.py` knows the doc_id of what it built, not
        which queue it came from, and a doc_id that exists in only one of them
        makes the other update a no-op.
        """
        moved = 0
        for table in self.conf.tables.values():
            result = (
                self.client.table(table)
                .update({"status": BUILT, "built_at": _now()})
                .eq("doc_id", doc_id)
                .eq("status", DOWNLOADED)
                .execute()
            )
            moved += len(result.data or [])
        return moved

    # -- the work -----------------------------------------------------------

    def sync_one(self, doc_type: str, row: Dict[str, Any], dry_run: bool = False) -> SyncOutcome:
        doc_id = row.get("doc_id", "?")
        storage_path = row["storage_path"]

        def failed(detail: str) -> SyncOutcome:
            if not dry_run:
                self._mark(doc_type, row["id"], FAILED, error=detail[:4000])
            return SyncOutcome(doc_id, FAILED, detail, doc_type)

        try:
            content = self.client.storage.from_(self.conf.bucket).download(storage_path)
        except Exception as e:
            return failed(f"download failed: {e}")

        digest = hashlib.sha256(content).hexdigest()
        if digest != row["sha256"]:
            return failed(f"sha256 mismatch — expected {row['sha256'][:12]}…, got {digest[:12]}…")

        try:
            file_name, blob, lines, notes = prepare_download(
                doc_type, content, storage_path,
                meeting_eligible_min_chars=self.conf.meeting_eligible_min_chars,
            )
        except UnicodeDecodeError as e:
            return failed(f"file is not valid UTF-8: {e}")
        except ValueError as e:
            return failed(f"cannot read as a {doc_type} chunk file: {e}")

        report = validate_lines(lines, path=storage_path)
        if not report.ok:
            # The producer can read `error` on their own row, so put the actual
            # failing lines there rather than making them ask what went wrong.
            return failed(f"{len(report.errors)} schema error(s): " + " | ".join(report.errors[:MAX_LISTED]))

        for note in notes:
            logger.info(f"{doc_id}: {note}")
        for warning in report.warnings:
            logger.warning(f"{doc_id}: {warning}")

        # The table already decided the folder; this only surfaces a file that
        # went into the wrong queue, which no longer has any other symptom.
        claimed = infer_doc_type(doc.source_kind for doc in report.docs.values())
        if claimed and claimed != doc_type:
            logger.warning(
                f"{doc_id}: uploaded to the {doc_type} queue but its records say {claimed} — "
                f"filing it as {doc_type}"
            )

        dest = Path(self.conf.dest_root) / doc_type / file_name
        if dry_run:
            return SyncOutcome(doc_id, DOWNLOADED, f"would write {dest} ({report.n_records} records)", doc_type)

        _write_atomically(dest, blob)
        self._mark(doc_type, row["id"], DOWNLOADED, downloaded_at=_now(), error=None)
        return SyncOutcome(doc_id, DOWNLOADED, f"{dest} ({report.n_records} records)", doc_type)

    def run(self, limit: int = 0, dry_run: bool = False) -> List[SyncOutcome]:
        pending = self.fetch_pending(limit=limit)
        if not pending:
            logger.info("Nothing pending.")
            return []

        by_type = {t: sum(1 for d, _ in pending if d == t) for t in sorted({d for d, _ in pending})}
        logger.info(f"{len(pending)} pending upload(s): " + ", ".join(f"{n} {t}" for t, n in by_type.items()))

        outcomes = []
        for doc_type, row in pending:
            outcome = self.sync_one(doc_type, row, dry_run=dry_run)
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

    try:
        meeting_eligible_min_chars = int(
            os.getenv("MEETING_ELIGIBLE_MIN_CHARS", str(DEFAULT_ELIGIBLE_MIN_CHARS))
        )
    except ValueError:
        raise RuntimeError("MEETING_ELIGIBLE_MIN_CHARS must be a whole number")

    return SyncConfig(
        url=url,
        service_key=service_key,
        bucket=os.getenv("SUPABASE_BUCKET", "chunks"),
        dest_root=dest_root or os.getenv("CHUNKS_DEST_DIR", "chunks_data"),
        tables={
            "paper": os.getenv("SUPABASE_PAPER_TABLE", DEFAULT_TABLES["paper"]),
            "meeting": os.getenv("SUPABASE_MEETING_TABLE", DEFAULT_TABLES["meeting"]),
        },
        meeting_eligible_min_chars=meeting_eligible_min_chars,
    )
