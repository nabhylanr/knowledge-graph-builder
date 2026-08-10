import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def participants_hash(description_ids: List[str]) -> str:
    """Hash of the sorted participant Description ids — §3.2's `participants_hash`,
    and the SQLite `contradiction_clusters.cluster_key`. Same hash scheme as
    `pair_key`: sha1 over ids joined with a separator unlikely to collide."""
    return hashlib.sha1("\x1f".join(sorted(description_ids)).encode("utf-8")).hexdigest()


def pair_key(description_id_a: str, description_id_b: str) -> str:
    """
    Deterministic key for an unordered Description pair. Callers must already
    pass ids in normalised (sorted) order — this only hashes them, it does not
    sort, so a caller-side ordering bug would silently produce two rows for
    the same pair instead of one.
    """
    return hashlib.sha1(f"{description_id_a}\x1f{description_id_b}".encode("utf-8")).hexdigest()


@dataclass
class CandidateRow:
    description_id_a: str
    description_id_b: str
    source_id_a: str
    source_id_b: str
    topic_a: str
    topic_b: str
    type_name: str
    strategy: str  # "exact" | "knn" | "both"
    similarity: Optional[float]

    def __post_init__(self):
        if self.description_id_a >= self.description_id_b:
            raise ValueError(
                f"CandidateRow ids must be pre-sorted (a < b); got "
                f"{self.description_id_a!r} >= {self.description_id_b!r}"
            )
        if self.strategy not in ("exact", "knn", "both"):
            raise ValueError(f"invalid strategy: {self.strategy!r}")

    @property
    def pair_key(self) -> str:
        return pair_key(self.description_id_a, self.description_id_b)


@dataclass
class GateInputRow:
    """One row of `rows_needing_gate()` output — everything G1-G5 need, plus
    whatever NLI verdict (if any) is already cached from a previous run."""
    pair_key: str
    description_id_a: str
    description_id_b: str
    source_id_a: str
    source_id_b: str
    topic_a: str
    topic_b: str
    nli_label: Optional[str]
    nli_confidence: Optional[float]
    nli_cache_key: Optional[str]


@dataclass
class GateResult:
    pair_key: str
    gate_status: str  # "passed" | "rejected"
    gate_reason: Optional[str]
    gate_version: str
    gate_checked_at: str
    nli_label: Optional[str]
    nli_confidence: Optional[float]
    nli_cache_key: Optional[str]


# Added on top of the stage-1 schema below (which already reserves gate_status/
# gate_reason). SQLite has no `ADD COLUMN IF NOT EXISTS`, so `_ensure_gate_columns`
# checks `PRAGMA table_info` first — idempotent on both a fresh DB (harmless
# no-op right after CREATE TABLE) and a pre-existing stage-1-only DB.
_GATE_COLUMNS = {
    "gate_version": "TEXT",
    "gate_checked_at": "TEXT",
    "nli_label": "TEXT",
    "nli_confidence": "REAL",
    "nli_cache_key": "TEXT",
}


def _ensure_gate_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(candidate_pairs)")}
    for name, col_type in _GATE_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE candidate_pairs ADD COLUMN {name} {col_type}")
    conn.commit()


@dataclass
class PassedPairRow:
    """One `gate_status = 'passed'` row, plus whatever supersession verdict (if
    any) is already cached from a previous stage-3 run."""
    pair_key: str
    description_id_a: str
    description_id_b: str
    source_id_a: str
    source_id_b: str
    topic_a: str
    topic_b: str
    type_name: str
    supersession_result: Optional[str]        # 'yes' | 'no' | 'anti_cycle_blocked' | 'not_evaluable' | None
    supersession_basis: Optional[str]
    supersession_confidence: Optional[float]
    supersession_cache_key: Optional[str]


@dataclass
class SupersessionResultRow:
    pair_key: str
    supersession_result: str
    supersession_basis: Optional[str]
    supersession_reason: Optional[str]
    supersession_confidence: Optional[float]
    supersession_cache_key: Optional[str]  # None when the verdict must NOT be cached (e.g. a failed LLM call)
    classification_pipeline_version: str
    classification_checked_at: str


@dataclass
class ClusterRecord:
    """
    One row of `contradiction_clusters` — SQLite's own record of a cluster's
    classification outcome, independent of whether Neo4j still has the node
    (see docs/conflict_pipeline.md §5, "an external index is a cache, not the
    source of truth"). Stores the FULL verdict, not just resolution_type, so a
    cache-hit-but-Neo4j-missing cluster (e.g. after a Neo4j wipe) can be
    rewritten without re-calling the LLM.
    """
    cluster_key: str  # participants_hash
    participant_ids: List[str]
    contradiction_node_id: Optional[str]      # None unless outcome == 'written'
    resolution_type: Optional[str]            # None unless outcome == 'written'
    summary: Optional[str]
    scope_conditions: Optional[str]
    confidence: Optional[float]
    evidence_used: List[str] = field(default_factory=list)
    positions: List[dict] = field(default_factory=list)  # [{"description_id": ..., "position": ...}]
    outcome: str = "written"                  # 'written' | 'insufficient_evidence' | 'not_a_conflict'
    # Reused for both insufficient_evidence's and not_a_conflict's free-text reason —
    # same physical column, no schema field added just for the new outcome (the
    # prompt still asks the model for a distinct `not_a_conflict_reason`, mapped
    # into this same field at the classifier.py boundary — see _not_a_conflict()).
    insufficient_evidence_reason: Optional[str] = None
    cache_key: Optional[str] = None           # None when not cacheable (e.g. failed LLM call)
    pipeline_version: str = ""


_CLASSIFICATION_COLUMNS = {
    "supersession_result": "TEXT",
    "supersession_basis": "TEXT",
    "supersession_reason": "TEXT",
    "supersession_confidence": "REAL",
    "supersession_cache_key": "TEXT",
    "classification_pipeline_version": "TEXT",
    "classification_checked_at": "TEXT",
}

_CONTRADICTION_CLUSTERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS contradiction_clusters (
    cluster_key                  TEXT PRIMARY KEY,
    participant_ids               TEXT NOT NULL,
    contradiction_node_id         TEXT,
    resolution_type               TEXT,
    summary                        TEXT,
    scope_conditions                TEXT,
    confidence                       REAL,
    evidence_used                     TEXT,
    positions                          TEXT,
    outcome                             TEXT NOT NULL CHECK(outcome IN ('written','insufficient_evidence','not_a_conflict')),
    insufficient_evidence_reason         TEXT,
    cache_key                             TEXT,
    pipeline_version                       TEXT NOT NULL,
    created_at                              TEXT NOT NULL,
    last_seen_at                             TEXT NOT NULL
);
"""


def _migrate_contradiction_clusters_outcome_check(conn: sqlite3.Connection) -> None:
    """
    SQLite CHECK constraints can't be altered in place — `CREATE TABLE IF NOT
    EXISTS` is a no-op against an existing table, so a pre-existing database
    file (created before 'not_a_conflict' was added) would keep the OLD
    CHECK(outcome IN ('written','insufficient_evidence')) forever, and every
    not_a_conflict upsert would raise a CHECK constraint violation at runtime.
    Standard SQLite migration: rename the old table aside, let the schema
    script below create the new one with the updated CHECK, copy the rows
    across, drop the old one. No-op on a fresh DB (table doesn't exist yet —
    the CREATE TABLE IF NOT EXISTS already has the new constraint) or an
    already-migrated one.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='contradiction_clusters'"
    ).fetchone()
    if row is None or "not_a_conflict" in row[0]:
        return
    conn.execute("ALTER TABLE contradiction_clusters RENAME TO contradiction_clusters_old")
    conn.executescript(_CONTRADICTION_CLUSTERS_SCHEMA)
    conn.execute("INSERT INTO contradiction_clusters SELECT * FROM contradiction_clusters_old")
    conn.execute("DROP TABLE contradiction_clusters_old")
    conn.commit()
    logger.info("Migrated contradiction_clusters.outcome CHECK constraint to include 'not_a_conflict'.")


def _ensure_classification_schema(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(candidate_pairs)")}
    for name, col_type in _CLASSIFICATION_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE candidate_pairs ADD COLUMN {name} {col_type}")
    _migrate_contradiction_clusters_outcome_check(conn)
    conn.executescript(_CONTRADICTION_CLUSTERS_SCHEMA)
    conn.commit()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_pairs (
    pair_key                    TEXT PRIMARY KEY,
    description_id_a            TEXT NOT NULL,
    description_id_b            TEXT NOT NULL,
    source_id_a                 TEXT NOT NULL,
    source_id_b                 TEXT NOT NULL,
    topic_a                     TEXT NOT NULL,
    topic_b                     TEXT NOT NULL,
    type_name                   TEXT NOT NULL,
    strategy                    TEXT NOT NULL CHECK(strategy IN ('exact','knn','both')),
    similarity                  REAL,
    created_at                  TEXT NOT NULL,
    pipeline_version             TEXT NOT NULL,
    last_seen_at                 TEXT NOT NULL,
    last_seen_pipeline_version    TEXT NOT NULL,
    -- reserved for later stages (gate filtering / LLM classification), so
    -- attaching outcomes there needs no migration:
    gate_status                  TEXT,
    gate_reason                   TEXT,
    classification_result         TEXT,
    classified_at                  TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidate_pairs_type     ON candidate_pairs(type_name);
CREATE INDEX IF NOT EXISTS idx_candidate_pairs_strategy ON candidate_pairs(strategy);
"""

_UPSERT = """
INSERT INTO candidate_pairs (
    pair_key, description_id_a, description_id_b, source_id_a, source_id_b,
    topic_a, topic_b, type_name, strategy, similarity,
    created_at, pipeline_version, last_seen_at, last_seen_pipeline_version
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(pair_key) DO UPDATE SET
    strategy = excluded.strategy,
    similarity = excluded.similarity,
    last_seen_at = excluded.last_seen_at,
    last_seen_pipeline_version = excluded.last_seen_pipeline_version
"""


class CandidateStore:
    """
    SQLite-backed store for stage-1 candidate pairs. Deliberately outside
    Neo4j — these are working state for the conflict-detection pipeline, not
    knowledge, per docs/conflict_ontology.md's separation between the graph
    (facts) and this pass's intermediate output.

    Upsert is keyed by `pair_key` (sorted-id hash) alone — NOT pipeline_version
    — so re-running (even after a config change) updates a pair's row in place
    rather than creating a parallel duplicate for the same pair. `created_at`
    and `pipeline_version` are the FIRST-seen stamp and are never overwritten
    by an upsert; `last_seen_at` / `last_seen_pipeline_version` are updated on
    every run so "first seen under v1, still present under v3" is answerable
    without per-version row history. Full classification-decision history
    belongs in a separate audit table in stage 3, not here.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        _ensure_gate_columns(self._conn)
        _ensure_classification_schema(self._conn)

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def upsert_candidates(self, rows: List[CandidateRow], pipeline_version: str) -> int:
        """Idempotent upsert. Returns the number of rows passed in (not just newly-inserted)."""
        now = datetime.now(timezone.utc).isoformat()
        params = [
            (
                r.pair_key, r.description_id_a, r.description_id_b, r.source_id_a, r.source_id_b,
                r.topic_a, r.topic_b, r.type_name, r.strategy, r.similarity,
                now, pipeline_version, now, pipeline_version,
            )
            for r in rows
        ]
        with self._conn:
            self._conn.executemany(_UPSERT, params)
        logger.info(f"Upserted {len(rows)} candidate pairs into {self.db_path}")
        return len(rows)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM candidate_pairs").fetchone()[0]

    def count_by_strategy(self) -> dict:
        rows = self._conn.execute(
            "SELECT strategy, COUNT(*) FROM candidate_pairs GROUP BY strategy"
        ).fetchall()
        return dict(rows)

    def rows_needing_gate(self, gate_version: str) -> List[GateInputRow]:
        """
        Rows not yet gated under `gate_version` — either never gated
        (`gate_status IS NULL`) or gated under a different version string.
        `!=` rather than an ordered "older than" comparison: `gate_version` is
        a free-form config stamp with no orderable semantics, so not-equal is
        the only honest check (matches `pipeline_version`'s own convention).
        """
        cur = self._conn.execute(
            """
            SELECT pair_key, description_id_a, description_id_b, source_id_a, source_id_b,
                   topic_a, topic_b, nli_label, nli_confidence, nli_cache_key
            FROM candidate_pairs
            WHERE gate_status IS NULL OR gate_version IS NULL OR gate_version != ?
            """,
            (gate_version,),
        )
        return [GateInputRow(*row) for row in cur.fetchall()]

    def update_gate_results(self, results: List[GateResult]) -> int:
        """In-place update only — never deletes a row. Rejected pairs stay
        queryable and re-runnable, per the gate-filtering contract."""
        with self._conn:
            self._conn.executemany(
                """
                UPDATE candidate_pairs
                SET gate_status = ?, gate_reason = ?, gate_version = ?, gate_checked_at = ?,
                    nli_label = ?, nli_confidence = ?, nli_cache_key = ?
                WHERE pair_key = ?
                """,
                [
                    (r.gate_status, r.gate_reason, r.gate_version, r.gate_checked_at,
                     r.nli_label, r.nli_confidence, r.nli_cache_key, r.pair_key)
                    for r in results
                ],
            )
        logger.info(f"Updated gate results for {len(results)} candidate pairs")
        return len(results)

    def count_by_gate_status(self) -> dict:
        rows = self._conn.execute(
            "SELECT gate_status, COUNT(*) FROM candidate_pairs GROUP BY gate_status"
        ).fetchall()
        return dict(rows)

    def count_by_gate_reason(self) -> dict:
        rows = self._conn.execute(
            "SELECT gate_reason, COUNT(*) FROM candidate_pairs WHERE gate_reason IS NOT NULL GROUP BY gate_reason"
        ).fetchall()
        return dict(rows)

    # ---- stage 3: classification ------------------------------------------

    def get_passed_pairs(self) -> List[PassedPairRow]:
        """Every currently gate-passed pair, plus any cached supersession
        verdict. Read fresh every run — if a gate re-run later flips a pair
        away from 'passed', it simply stops appearing here."""
        cur = self._conn.execute(
            """
            SELECT pair_key, description_id_a, description_id_b, source_id_a, source_id_b,
                   topic_a, topic_b, type_name,
                   supersession_result, supersession_basis, supersession_confidence, supersession_cache_key
            FROM candidate_pairs WHERE gate_status = 'passed'
            """
        )
        return [PassedPairRow(*row) for row in cur.fetchall()]

    def update_supersession_results(self, results: List[SupersessionResultRow]) -> int:
        with self._conn:
            self._conn.executemany(
                """
                UPDATE candidate_pairs
                SET supersession_result = ?, supersession_basis = ?, supersession_reason = ?,
                    supersession_confidence = ?, supersession_cache_key = ?,
                    classification_pipeline_version = ?, classification_checked_at = ?
                WHERE pair_key = ?
                """,
                [
                    (r.supersession_result, r.supersession_basis, r.supersession_reason,
                     r.supersession_confidence, r.supersession_cache_key,
                     r.classification_pipeline_version, r.classification_checked_at, r.pair_key)
                    for r in results
                ],
            )
        logger.info(f"Updated supersession results for {len(results)} candidate pairs")
        return len(results)

    def count_by_supersession_result(self) -> dict:
        rows = self._conn.execute(
            "SELECT supersession_result, COUNT(*) FROM candidate_pairs "
            "WHERE supersession_result IS NOT NULL GROUP BY supersession_result"
        ).fetchall()
        return dict(rows)

    def count_supersession_by_basis(self) -> dict:
        rows = self._conn.execute(
            "SELECT supersession_basis, COUNT(*) FROM candidate_pairs "
            "WHERE supersession_basis IS NOT NULL GROUP BY supersession_basis"
        ).fetchall()
        return dict(rows)

    def get_cluster_records(self) -> Dict[str, ClusterRecord]:
        """All existing cluster records, keyed by cluster_key — fetched once per
        run rather than one query per cluster."""
        cur = self._conn.execute(
            """
            SELECT cluster_key, participant_ids, contradiction_node_id, resolution_type,
                   summary, scope_conditions, confidence, evidence_used, positions,
                   outcome, insufficient_evidence_reason, cache_key, pipeline_version
            FROM contradiction_clusters
            """
        )
        out = {}
        for row in cur.fetchall():
            (cluster_key, participant_ids_json, contradiction_node_id, resolution_type,
             summary, scope_conditions, confidence, evidence_used_json, positions_json,
             outcome, insufficient_evidence_reason, cache_key, pipeline_version) = row
            out[cluster_key] = ClusterRecord(
                cluster_key=cluster_key,
                participant_ids=json.loads(participant_ids_json),
                contradiction_node_id=contradiction_node_id,
                resolution_type=resolution_type,
                summary=summary,
                scope_conditions=scope_conditions,
                confidence=confidence,
                evidence_used=json.loads(evidence_used_json) if evidence_used_json else [],
                positions=json.loads(positions_json) if positions_json else [],
                outcome=outcome,
                insufficient_evidence_reason=insufficient_evidence_reason,
                cache_key=cache_key,
                pipeline_version=pipeline_version,
            )
        return out

    def upsert_cluster_record(self, record: ClusterRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO contradiction_clusters (
                    cluster_key, participant_ids, contradiction_node_id, resolution_type,
                    summary, scope_conditions, confidence, evidence_used, positions,
                    outcome, insufficient_evidence_reason, cache_key, pipeline_version,
                    created_at, last_seen_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(cluster_key) DO UPDATE SET
                    participant_ids = excluded.participant_ids,
                    contradiction_node_id = excluded.contradiction_node_id,
                    resolution_type = excluded.resolution_type,
                    summary = excluded.summary,
                    scope_conditions = excluded.scope_conditions,
                    confidence = excluded.confidence,
                    evidence_used = excluded.evidence_used,
                    positions = excluded.positions,
                    outcome = excluded.outcome,
                    insufficient_evidence_reason = excluded.insufficient_evidence_reason,
                    cache_key = excluded.cache_key,
                    pipeline_version = excluded.pipeline_version,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    record.cluster_key, json.dumps(sorted(record.participant_ids)),
                    record.contradiction_node_id, record.resolution_type,
                    record.summary, record.scope_conditions, record.confidence,
                    json.dumps(record.evidence_used), json.dumps(record.positions),
                    record.outcome, record.insufficient_evidence_reason, record.cache_key,
                    record.pipeline_version, now, now,
                ),
            )

    def count_clusters_by_resolution_type(self) -> dict:
        rows = self._conn.execute(
            "SELECT resolution_type, COUNT(*) FROM contradiction_clusters "
            "WHERE resolution_type IS NOT NULL GROUP BY resolution_type"
        ).fetchall()
        return dict(rows)

    def count_clusters_by_outcome(self) -> dict:
        rows = self._conn.execute(
            "SELECT outcome, COUNT(*) FROM contradiction_clusters GROUP BY outcome"
        ).fetchall()
        return dict(rows)
