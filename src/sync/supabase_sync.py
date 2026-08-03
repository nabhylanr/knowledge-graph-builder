"""
Pulls chunk files a producer uploaded to Supabase down onto this machine.

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
from typing import Any, Dict, List, Optional

from src.ingestion.validate import MAX_LISTED, validate_lines
from src.utils.logger import get_logger

logger = get_logger(__name__)

PENDING = "pending"
DOWNLOADED = "downloaded"
BUILT = "built"
FAILED = "failed"


@dataclass
class SyncConfig:
    url: str
    service_key: str
    bucket: str = "chunks"
    dest_root: str = "chunks_data"
    table: str = "chunk_uploads"


@dataclass
class SyncOutcome:
    doc_id: str
    status: str
    detail: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_destination(dest_root: str, storage_path: str) -> Path:
    """
    Map a bucket path to a local path, refusing anything that would escape
    `dest_root`. `storage_path` comes from a row a producer wrote, so it is
    untrusted input — "../../.ssh/authorized_keys" is a valid string.
    """
    rel = PurePosixPath(storage_path)
    if rel.is_absolute() or any(part in ("..", "") for part in rel.parts):
        raise ValueError(f"unsafe storage_path: {storage_path!r}")
    return Path(dest_root).joinpath(*rel.parts)


def _write_atomically(dest: Path, content: bytes) -> None:
    """Write via a temp file + rename so an interrupted sync never leaves a
    half-written .jsonl that looks complete to the next build."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(content)
    os.replace(tmp, dest)


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
            dest = _safe_destination(self.conf.dest_root, storage_path)
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

        if dry_run:
            return SyncOutcome(doc_id, DOWNLOADED, f"would write {dest} ({report.n_records} records)")

        _write_atomically(dest, content)
        self._mark(row["id"], DOWNLOADED, downloaded_at=_now(), error=None)
        return SyncOutcome(doc_id, DOWNLOADED, f"{dest} ({report.n_records} records)")

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
    return SyncConfig(
        url=url,
        service_key=service_key,
        bucket=os.getenv("SUPABASE_BUCKET", "chunks"),
        dest_root=dest_root or os.getenv("CHUNKS_DEST_DIR", "chunks_data"),
    )
