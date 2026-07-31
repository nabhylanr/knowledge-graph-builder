from typing import Dict, List

from src.graph.knowledge_graph import KnowledgeGraph

# Method/Dataset/Experiment Descriptions describe what was done or assumed, not
# what was found — never conflict participants themselves (§3.1), but exactly
# the scope context §4.3(a) needs to judge whether an opposition is explained
# by a difference in method, dataset, population, setting, etc.
SCOPE_CONTEXT_TYPES = ["Method", "Dataset", "Experiment"]


def fetch_source_years(kg: KnowledgeGraph, source_ids: List[str]) -> Dict[str, int]:
    """Batched `Source.year` fetch for every distinct source_id referenced by
    the pairs being processed this run. `year` is stored as a string
    (§2.0 prerequisite) — parsed here via toInteger(); a Source with no
    parseable year is simply absent from the returned dict."""
    if not source_ids:
        return {}
    rows = kg.query(
        "MATCH (s:Source) WHERE s.id IN $ids AND s.year IS NOT NULL "
        "RETURN s.id AS id, toInteger(s.year) AS year",
        params={"ids": source_ids},
    )
    return {r["id"]: r["year"] for r in rows if r["year"] is not None}


def fetch_description_fields(kg: KnowledgeGraph, description_ids: List[str]) -> Dict[str, dict]:
    """Batched fetch of {id: {text, topicName, source_id, typeName}} — one
    query for every Description this run needs, not one per pair/cluster."""
    if not description_ids:
        return {}
    rows = kg.query(
        "MATCH (d:Description) WHERE d.id IN $ids "
        "RETURN d.id AS id, d.text AS text, d.topicName AS topicName, "
        "d.source_id AS source_id, d.typeName AS typeName",
        params={"ids": description_ids},
    )
    return {r["id"]: r for r in rows}


def fetch_scope_context(kg: KnowledgeGraph, source_ids: List[str]) -> Dict[str, List[dict]]:
    """
    Batched fetch of Method/Dataset/Experiment Description texts sharing
    source_id, for every source_id referenced by ANY pair/cluster processed
    this run, combined into ONE query — better than "one query per cluster"
    (the task's stated ceiling) since clusters commonly share source_ids.
    Returns {source_id: [{"typeName": ..., "text": ...}, ...]}.
    """
    if not source_ids:
        return {}
    rows = kg.query(
        "MATCH (d:Description) WHERE d.source_id IN $ids AND d.typeName IN $scope_types "
        "RETURN d.source_id AS source_id, d.typeName AS typeName, d.text AS text",
        params={"ids": source_ids, "scope_types": SCOPE_CONTEXT_TYPES},
    )
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(r["source_id"], []).append({"typeName": r["typeName"], "text": r["text"]})
    return out
