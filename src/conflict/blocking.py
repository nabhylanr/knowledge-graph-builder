from typing import Dict, List

from src.conflict.candidate_store import CandidateRow, CandidateStore
from src.conflict.config import BlockingConfig
from src.graph.knowledge_graph import KnowledgeGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)

# S1 groups Descriptions by (typeName, topicName); without this composite index
# Neo4j falls back to a full node scan + join, which is the O(n^2) behaviour the
# task rules out at "tens of thousands of Descriptions" scale. A schema index is
# infrastructure, not knowledge content, so it's in scope despite this module
# otherwise being read-only against the graph.
_SCHEMA_INDEX = (
    "CREATE INDEX description_topic_type IF NOT EXISTS "
    "FOR (d:Description) ON (d.typeName, d.topicName)"
)

_S1_QUERY = """
MATCH (a:Description), (b:Description)
WHERE a.typeName = b.typeName
  AND a.topicName = b.topicName
  AND a.typeName IN $allowed_types
  AND a.id < b.id
RETURN a.id AS id_a, b.id AS id_b,
       a.topicName AS topic_a, b.topicName AS topic_b,
       a.typeName AS type_name,
       a.source_id AS source_id_a, b.source_id AS source_id_b
"""

# One vector-index lookup per Description (via CALL subquery, batched
# server-side) rather than an all-pairs scan — O(N * k), not O(N^2). Neo4j
# 5.9+ `CALL (d) { ... }` variable-scoped subquery syntax; docker-compose.yml
# pins 5.26, so no fallback to the older `CALL { WITH d ... }` form is needed.
_S2_QUERY = """
MATCH (d:Description)
WHERE d.typeName IN $allowed_types AND d.embedding IS NOT NULL
CALL (d) {
  CALL db.index.vector.queryNodes($index_name, $k_plus_one, d.embedding)
  YIELD node AS neighbor, score
  WHERE neighbor.id <> d.id
    AND neighbor.typeName = d.typeName
    AND ($min_similarity IS NULL OR score >= $min_similarity)
  RETURN neighbor, score
}
RETURN d.id AS id_from, neighbor.id AS id_to, score,
       d.topicName AS topic_from, neighbor.topicName AS topic_to,
       d.typeName AS type_name,
       d.source_id AS source_id_from, neighbor.source_id AS source_id_to
"""


def _ensure_schema_index(kg: KnowledgeGraph) -> None:
    kg.query(_SCHEMA_INDEX)


def _s1_exact_pairs(kg: KnowledgeGraph, conf: BlockingConfig) -> List[dict]:
    return kg.query(_S1_QUERY, params={"allowed_types": conf.allowed_types})


def _s2_knn_pairs(kg: KnowledgeGraph, conf: BlockingConfig) -> List[dict]:
    rows = kg.query(
        _S2_QUERY,
        params={
            "allowed_types": conf.allowed_types,
            "index_name": conf.description_index_name,
            "k_plus_one": conf.k + 1,  # a node's own vector is usually its own top match
            "min_similarity": conf.min_similarity,
        },
    )
    # Collapse directed (id_from -> id_to) results into unordered pairs, keeping
    # the higher score if both directions surfaced the same pair (each node's
    # own top-k lookup is independent, so this is common, not an edge case).
    merged: Dict[tuple, dict] = {}
    for r in rows:
        a, b = r["id_from"], r["id_to"]
        if a == b:
            continue
        if a < b:
            id_a, id_b, topic_a, topic_b, source_a, source_b = (
                a, b, r["topic_from"], r["topic_to"], r["source_id_from"], r["source_id_to"]
            )
        else:
            id_a, id_b, topic_a, topic_b, source_a, source_b = (
                b, a, r["topic_to"], r["topic_from"], r["source_id_to"], r["source_id_from"]
            )
        key = (id_a, id_b)
        existing = merged.get(key)
        if existing is None or r["score"] > existing["score"]:
            merged[key] = {
                "id_a": id_a, "id_b": id_b,
                "topic_a": topic_a, "topic_b": topic_b,
                "type_name": r["type_name"],
                "source_id_a": source_a, "source_id_b": source_b,
                "score": r["score"],
            }
    return list(merged.values())


def _merge(s1_rows: List[dict], s2_rows: List[dict]) -> List[CandidateRow]:
    by_key: Dict[tuple, CandidateRow] = {}

    for r in s1_rows:
        key = (r["id_a"], r["id_b"])
        by_key[key] = CandidateRow(
            description_id_a=r["id_a"], description_id_b=r["id_b"],
            source_id_a=r["source_id_a"], source_id_b=r["source_id_b"],
            topic_a=r["topic_a"], topic_b=r["topic_b"],
            type_name=r["type_name"], strategy="exact", similarity=None,
        )

    for r in s2_rows:
        key = (r["id_a"], r["id_b"])
        if key in by_key:
            existing = by_key[key]
            by_key[key] = CandidateRow(
                description_id_a=existing.description_id_a, description_id_b=existing.description_id_b,
                source_id_a=existing.source_id_a, source_id_b=existing.source_id_b,
                topic_a=existing.topic_a, topic_b=existing.topic_b,
                type_name=existing.type_name, strategy="both", similarity=r["score"],
            )
        else:
            by_key[key] = CandidateRow(
                description_id_a=r["id_a"], description_id_b=r["id_b"],
                source_id_a=r["source_id_a"], source_id_b=r["source_id_b"],
                topic_a=r["topic_a"], topic_b=r["topic_b"],
                type_name=r["type_name"], strategy="knn", similarity=r["score"],
            )

    return list(by_key.values())


def generate_candidates(kg: KnowledgeGraph, conf: BlockingConfig, store: CandidateStore) -> dict:
    """
    Stage 1 of the whole-KB conflict pass (docs/conflict_ontology.md): produces
    the deduplicated candidate-pair set (S1 exact-match union S2 kNN) and
    upserts it into `store`. No LLM calls, no writes to graph structure beyond
    the schema/vector index. Read-only against Description content.
    """
    _ensure_schema_index(kg)

    s1_rows = _s1_exact_pairs(kg, conf)
    s2_rows = _s2_knn_pairs(kg, conf)
    candidates = _merge(s1_rows, s2_rows)

    store.upsert_candidates(candidates, pipeline_version=conf.pipeline_version)

    stats = {
        "s1_pairs": len(s1_rows),
        "s2_pairs": len(s2_rows),
        "candidates_total": len(candidates),
        "candidates_exact_only": sum(1 for c in candidates if c.strategy == "exact"),
        "candidates_knn_only": sum(1 for c in candidates if c.strategy == "knn"),
        "candidates_both": sum(1 for c in candidates if c.strategy == "both"),
    }
    logger.info(f"Candidate generation complete: {stats}")
    return stats
