import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import networkx as nx

from src.conflict.candidate_store import (
    CandidateStore,
    ClusterRecord,
    PassedPairRow,
    SupersessionResultRow,
    participants_hash,
)
from src.conflict.classification_client import ClassificationClient
from src.conflict.config import ClassificationConfig, HAS_CONTRADICTION_TYPE, SUPERSEDES_TYPE
from src.conflict.context import fetch_description_fields, fetch_scope_context, fetch_source_years
from src.graph.knowledge_graph import KnowledgeGraph
from src.prompts.cluster_classification import format_participants_block
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---- cache keys -------------------------------------------------------------

def _supersession_cache_key(newer_id: str, older_id: str, newer_text: str, older_text: str,
                             model: str, prompt_version: str) -> str:
    h = hashlib.sha1()
    for part in (newer_id, older_id, newer_text, older_text, model, prompt_version):
        h.update((part or "").encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _cluster_cache_key(cluster_key: str, model: str, prompt_version: str, pipeline_version: str) -> str:
    h = hashlib.sha1()
    for part in (cluster_key, model, prompt_version, pipeline_version):
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


# ---- Neo4j reads (batched) ---------------------------------------------------

def _check_supersedes_edges_exist(kg: KnowledgeGraph, pairs: List[Tuple[str, str]]):
    """Which (newer_id, older_id) pairs still have a live `supersedes` edge —
    one batched query for every cached 'yes' pair this run, not one per pair.
    Needed because a cached 'yes' in SQLite is not proof the edge survives in
    Neo4j (Neo4j is wiped/re-ingested regularly; SQLite persists across that —
    same failure mode amendment 1 fixes for Contradiction, applied here too)."""
    if not pairs:
        return set()
    rows = kg.query(
        f"""
        UNWIND $pairs AS p
        MATCH (newer:Description {{id: p.newer_id}})-[:{SUPERSEDES_TYPE}]->(older:Description {{id: p.older_id}})
        RETURN p.newer_id AS newer_id, p.older_id AS older_id
        """,
        params={"pairs": [{"newer_id": a, "older_id": b} for a, b in pairs]},
    )
    return {(r["newer_id"], r["older_id"]) for r in rows}


def _fetch_exact_hash_matches(kg: KnowledgeGraph, hashes: List[str]) -> Dict[str, tuple]:
    """AMENDMENT 1: the Neo4j-side half of the staleness check. A cluster may
    only be skipped when SQLite says 'already classified under this
    pipeline_version' AND a Contradiction with that exact participants_hash
    still exists here. One batched query for every candidate cluster hash."""
    if not hashes:
        return {}
    rows = kg.query(
        "MATCH (c:Contradiction) WHERE c.participants_hash IN $hashes "
        "RETURN c.participants_hash AS hash, c.id AS id, c.generated_at AS generated_at",
        params={"hashes": hashes},
    )
    return {r["hash"]: (r["id"], r["generated_at"]) for r in rows}


def _fetch_overlapping_contradictions(kg: KnowledgeGraph, description_ids: List[str]) -> Dict[str, List[tuple]]:
    """Identity resolution input: for every participant id across every
    cluster being WRITTEN this run (not skipped), which existing Contradiction
    node(s) does it already belong to. One batched query for the whole run."""
    if not description_ids:
        return {}
    rows = kg.query(
        f"MATCH (d:Description)-[:{HAS_CONTRADICTION_TYPE}]->(c:Contradiction) WHERE d.id IN $ids "
        "RETURN d.id AS description_id, c.id AS id, c.generated_at AS generated_at",
        params={"ids": description_ids},
    )
    out: Dict[str, List[tuple]] = {}
    for r in rows:
        out.setdefault(r["description_id"], []).append((r["id"], r["generated_at"]))
    return out


# ---- Neo4j writes -------------------------------------------------------------

def _delete_contradictions(kg: KnowledgeGraph, contradiction_ids: List[str]) -> None:
    """Used when a cluster's participants previously belonged to a Contradiction
    node but this run's outcome is SQLite-only (not_a_conflict,
    insufficient_evidence) — the old node must not survive orphaned, since its
    participant set no longer corresponds to a valid conflict per this run.
    Single non-transactional statement (unlike _write_cluster_tx): a plain
    DETACH DELETE is already atomic and idempotent, no multi-step reconciliation
    needed."""
    if not contradiction_ids:
        return
    kg.query(
        "MATCH (c:Contradiction) WHERE c.id IN $ids DETACH DELETE c",
        params={"ids": contradiction_ids},
    )


def _write_supersedes(kg: KnowledgeGraph, newer_id: str, older_id: str, basis: Optional[str],
                       reason: Optional[str], confidence: Optional[float], pipeline_version: str,
                       generated_at: str, allowed_types: List[str]) -> bool:
    """One atomic statement: anti-cycle check + write together (§3.4 constraint
    3), avoiding a separate check-then-write race. Returns False if a reverse
    `supersedes` edge already existed (nothing written) or endpoint types are
    invalid (§3.4 constraint 1, defensive — blocking should already guarantee
    this)."""
    result = kg.query(
        f"""
        MATCH (newer:Description {{id: $newer_id}}), (older:Description {{id: $older_id}})
        WHERE newer.typeName IN $allowed_types AND older.typeName IN $allowed_types
          AND newer.id <> older.id
          AND NOT EXISTS {{ (older)-[:{SUPERSEDES_TYPE}]->(newer) }}
        MERGE (newer)-[r:{SUPERSEDES_TYPE}]->(older)
        SET r.basis = $basis, r.reason = $reason, r.confidence = $confidence,
            r.pipeline_version = $pipeline_version, r.generated_at = $generated_at
        RETURN count(r) AS written
        """,
        params={
            "newer_id": newer_id, "older_id": older_id, "allowed_types": allowed_types,
            "basis": basis, "reason": reason, "confidence": confidence,
            "pipeline_version": pipeline_version, "generated_at": generated_at,
        },
    )
    return bool(result) and result[0]["written"] > 0


def _write_cluster_tx(tx, cluster_ids: List[str], keep_id: str, drop_ids: List[str],
                       properties: dict, participant_positions: Dict[str, str]) -> Optional[str]:
    """
    One Neo4j transaction per cluster (a partial write leaves a singleton
    Contradiction, which §3.3 forbids): delete merge-losing nodes, upsert the
    surviving node's properties, reconcile has_contradiction edges to the
    CURRENT participant set, then (AMENDMENT 2) delete the node if that
    reconciliation left it with fewer than 2 edges.
    """
    if drop_ids:
        tx.run("MATCH (c:Contradiction) WHERE c.id IN $drop_ids DETACH DELETE c", drop_ids=drop_ids)

    tx.run(
        """
        MERGE (c:Contradiction {id: $id})
        SET c.summary = $summary, c.resolution_type = $resolution_type,
            c.scope_conditions = $scope_conditions, c.confidence = $confidence,
            c.participants_hash = $participants_hash, c.generated_by = $generated_by,
            c.generated_at = $generated_at, c.pipeline_version = $pipeline_version,
            c.evidence_used = $evidence_used
        """,
        id=keep_id, **properties,
    )

    tx.run(
        f"""
        MATCH (c:Contradiction {{id: $id}})<-[r:{HAS_CONTRADICTION_TYPE}]-(d:Description)
        WHERE NOT d.id IN $cluster_ids
        DELETE r
        """,
        id=keep_id, cluster_ids=cluster_ids,
    )

    tx.run(
        f"""
        UNWIND $participants AS p
        MATCH (d:Description {{id: p.id}})
        MATCH (c:Contradiction {{id: $keep_id}})
        MERGE (d)-[r:{HAS_CONTRADICTION_TYPE}]->(c)
        SET r.position = p.position
        """,
        participants=[{"id": pid, "position": participant_positions.get(pid)} for pid in cluster_ids],
        keep_id=keep_id,
    )

    result = tx.run(
        f"MATCH (c:Contradiction {{id: $id}}) OPTIONAL MATCH (c)<-[r:{HAS_CONTRADICTION_TYPE}]-() RETURN count(r) AS n",
        id=keep_id,
    )
    n = result.single()["n"]
    if n < 2:
        tx.run("MATCH (c:Contradiction {id: $id}) DETACH DELETE c", id=keep_id)
        logger.warning(
            f"Contradiction {keep_id} fell below 2 participants after edge reconciliation "
            f"(n={n}) — deleted in the same transaction per §3.3 / amendment 2."
        )
        return None
    return keep_id


# ---- stage: supersession (§4.1) ----------------------------------------------

def _process_supersession(
    pairs: List[PassedPairRow], years: Dict[str, int], desc_fields: Dict[str, dict],
    scope_context: Dict[str, List[dict]], client: ClassificationClient, kg: KnowledgeGraph,
    conf: ClassificationConfig,
) -> Tuple[List[SupersessionResultRow], List[PassedPairRow]]:
    """
    Returns (sqlite_updates, pairs_for_clustering). Only pairs that resolve
    'yes' AND whose edge is confirmed written are excluded from clustering —
    'no', 'anti_cycle_blocked', and 'not_evaluable' all fall through to §4.2,
    per the confirmed reading of §4.1 (exactly two outcomes, no
    insufficient-evidence branch here).
    """
    now = datetime.now(timezone.utc).isoformat()
    updates: List[SupersessionResultRow] = []
    for_clustering: List[PassedPairRow] = []
    cached_yes_to_verify: List[tuple] = []  # (pair, newer_id, older_id, basis, confidence, cache_key)

    for pair in pairs:
        year_a = years.get(pair.source_id_a)
        year_b = years.get(pair.source_id_b)

        if year_a is None or year_b is None or year_a == year_b:
            # §4.1: not evaluable — no LLM call, always falls through to clustering.
            updates.append(SupersessionResultRow(
                pair_key=pair.pair_key, supersession_result="not_evaluable",
                supersession_basis=None, supersession_reason=None, supersession_confidence=None,
                supersession_cache_key=None, classification_pipeline_version=conf.pipeline_version,
                classification_checked_at=now,
            ))
            for_clustering.append(pair)
            continue

        if year_a > year_b:
            newer_id, older_id = pair.description_id_a, pair.description_id_b
        else:
            newer_id, older_id = pair.description_id_b, pair.description_id_a
        newer_fields = desc_fields.get(newer_id, {})
        older_fields = desc_fields.get(older_id, {})
        newer_text, older_text = newer_fields.get("text", ""), older_fields.get("text", "")
        newer_source, older_source = newer_fields.get("source_id"), older_fields.get("source_id")

        cache_key = _supersession_cache_key(
            newer_id, older_id, newer_text, older_text, conf.model, conf.supersession_prompt_version
        )

        if pair.supersession_cache_key == cache_key and pair.supersession_result is not None:
            if pair.supersession_result == "yes":
                cached_yes_to_verify.append(
                    (pair, newer_id, older_id, pair.supersession_basis, pair.supersession_confidence, cache_key)
                )
                continue  # resolved after the batched verification pass below
            updates.append(SupersessionResultRow(
                pair_key=pair.pair_key, supersession_result=pair.supersession_result,
                supersession_basis=pair.supersession_basis, supersession_reason=None,
                supersession_confidence=pair.supersession_confidence, supersession_cache_key=cache_key,
                classification_pipeline_version=conf.pipeline_version, classification_checked_at=now,
            ))
            for_clustering.append(pair)
            continue

        verdict = client.classify_supersession(
            older_text=older_text, older_year=years.get(older_source),
            older_context_texts=[c["text"] for c in scope_context.get(older_source, [])],
            newer_text=newer_text, newer_year=years.get(newer_source),
            newer_context_texts=[c["text"] for c in scope_context.get(newer_source, [])],
        )

        if verdict is None:
            # Confirmed reading: failure/uncertainty -> "no", continue to step 2.
            # Not cached — a failed call may be transient (mirrors gates.py's
            # ERROR_LABEL non-caching).
            result, basis, reason, confidence, this_cache_key = "no", None, None, None, None
        elif verdict.decision == "no":
            result, basis, reason, confidence, this_cache_key = "no", None, verdict.reason, verdict.confidence, cache_key
        else:
            written = _write_supersedes(
                kg, newer_id=newer_id, older_id=older_id, basis=verdict.basis, reason=verdict.reason,
                confidence=verdict.confidence, pipeline_version=conf.pipeline_version, generated_at=now,
                allowed_types=conf.allowed_types,
            )
            if written:
                result, basis, reason, confidence, this_cache_key = "yes", verdict.basis, verdict.reason, verdict.confidence, cache_key
            else:
                logger.warning(
                    f"Anti-cycle: a reverse supersedes edge ({older_id} -> {newer_id}) already exists — "
                    f"routing ({newer_id}, {older_id}) to clustering instead of writing a cycle (§3.4 constraint 3)."
                )
                result, basis, reason, confidence, this_cache_key = "anti_cycle_blocked", verdict.basis, verdict.reason, verdict.confidence, cache_key

        updates.append(SupersessionResultRow(
            pair_key=pair.pair_key, supersession_result=result, supersession_basis=basis,
            supersession_reason=reason, supersession_confidence=confidence, supersession_cache_key=this_cache_key,
            classification_pipeline_version=conf.pipeline_version, classification_checked_at=now,
        ))
        if result != "yes":
            for_clustering.append(pair)

    if cached_yes_to_verify:
        existing = _check_supersedes_edges_exist(kg, [(n, o) for (_, n, o, *_rest) in cached_yes_to_verify])
        for pair, newer_id, older_id, basis, confidence, cache_key in cached_yes_to_verify:
            if (newer_id, older_id) in existing:
                updates.append(SupersessionResultRow(
                    pair_key=pair.pair_key, supersession_result="yes", supersession_basis=basis,
                    supersession_reason=None, supersession_confidence=confidence, supersession_cache_key=cache_key,
                    classification_pipeline_version=conf.pipeline_version, classification_checked_at=now,
                ))
                continue
            # AMENDMENT 1's failure mode, applied consistently to supersedes:
            # SQLite says "yes, cached" but Neo4j lost the edge (e.g. a wipe).
            # Re-write it without a new LLM call.
            rewritten = _write_supersedes(
                kg, newer_id=newer_id, older_id=older_id, basis=basis, reason=None, confidence=confidence,
                pipeline_version=conf.pipeline_version, generated_at=now, allowed_types=conf.allowed_types,
            )
            if rewritten:
                logger.info(f"Re-wrote supersedes edge missing from Neo4j (cache hit, no new LLM call): {newer_id} -> {older_id}")
                updates.append(SupersessionResultRow(
                    pair_key=pair.pair_key, supersession_result="yes", supersession_basis=basis,
                    supersession_reason=None, supersession_confidence=confidence, supersession_cache_key=cache_key,
                    classification_pipeline_version=conf.pipeline_version, classification_checked_at=now,
                ))
            else:
                logger.warning(
                    f"Cached supersedes edge {newer_id} -> {older_id} was missing from Neo4j and could not "
                    f"be re-written (reverse edge now exists?) — routing to clustering."
                )
                updates.append(SupersessionResultRow(
                    pair_key=pair.pair_key, supersession_result="anti_cycle_blocked", supersession_basis=basis,
                    supersession_reason=None, supersession_confidence=confidence, supersession_cache_key=cache_key,
                    classification_pipeline_version=conf.pipeline_version, classification_checked_at=now,
                ))
                for_clustering.append(pair)

    return updates, for_clustering


# ---- stage: clustering (§4.2) -------------------------------------------------

def _build_clusters(pairs_for_clustering: List[PassedPairRow], max_cluster_size: int) -> List[List[str]]:
    """
    Clique-based clustering, not connected components: a connected component
    ({A,B} passed, {B,C} passed) does NOT mean {A,B,C} is one cluster — A and C
    were never compared, and a summary spanning both is meaningless (this
    replaced connected components specifically because it produced exactly
    that: a 5-participant cluster spanning three unrelated subjects). A set of
    Descriptions may form one cluster only if EVERY pair among them
    independently passed the gate — i.e. only if it's a clique.

    Enumerates ALL maximal cliques per component (nx.find_cliques — NP-hard in
    general, a total non-issue at this corpus's scale of a few hundred
    Descriptions post-gating; would need revisiting at a much denser graph).
    Overlapping maximal cliques are NOT deduplicated down to one: if A-B and
    A-C both passed but B-C did not, {A,B} and {A,C} are two genuinely
    separate conflicts (A opposes B about one thing, A opposes C about
    something else) — picking only one would silently discard a real conflict.
    A component that is already a single clique (this includes every ordinary
    pairwise case, size 2) produces exactly one cluster, same as before.

    `max_cluster_size` is a separate safety net applied AFTER clique
    enumeration, per clique — decomposition bounds a non-clique component, but
    a genuinely dense clique (every pair independently passed) could still
    exceed a size the classification prompt can reason about sensibly.
    Truncated deterministically (sorted ids, first N kept) and logged.
    """
    graph = nx.Graph()
    for pair in pairs_for_clustering:
        graph.add_edge(pair.description_id_a, pair.description_id_b)

    clusters: List[List[str]] = []
    for component in nx.connected_components(graph):
        if len(component) < 2:
            continue
        subgraph = graph.subgraph(component)
        cliques = [c for c in nx.find_cliques(subgraph) if len(c) >= 2]

        if not (len(cliques) == 1 and len(cliques[0]) == len(component)):
            logger.info(
                f"Component of size {len(component)} is not a single clique — "
                f"decomposed into {len(cliques)} maximal clique(s), sizes "
                f"{sorted((len(c) for c in cliques), reverse=True)}"
            )

        for clique in cliques:
            if len(clique) > max_cluster_size:
                kept = sorted(clique)[:max_cluster_size]
                dropped = sorted(set(clique) - set(kept))
                logger.warning(
                    f"Clique of size {len(clique)} exceeds max_cluster_size={max_cluster_size} "
                    f"— truncating to {len(kept)}, dropped: {dropped}"
                )
                clusters.append(kept)
            else:
                clusters.append(sorted(clique))

    return clusters


# ---- stage: cluster classification (§4.3) -------------------------------------

def _run_cluster_classification(cluster: List[str], client: ClassificationClient, conf: ClassificationConfig,
                                 desc_fields: Dict[str, dict], scope_context: Dict[str, List[dict]]) -> dict:
    """Assembles the prompt, calls the LLM, and enforces AMENDMENT 4 (unresolved
    requires scope was checkable) and the §3.2/§8 invariants in code — the
    prompt states the rules, but a malformed or spec-violating verdict must
    never reach Neo4j regardless of what the model actually returns."""
    participants = []
    any_context = False
    all_context = True
    for i, pid in enumerate(cluster, start=1):
        fields = desc_fields.get(pid, {})
        source_id = fields.get("source_id")
        context_texts = [c["text"] for c in scope_context.get(source_id, [])]
        if context_texts:
            any_context = True
        else:
            all_context = False
        participants.append({
            "index": i, "id": pid, "source_id": source_id,
            "topic": fields.get("topicName", ""), "text": fields.get("text", ""),
            "context_texts": context_texts,
        })

    verdict = client.classify_cluster(format_participants_block(participants), n=len(participants))

    def _insufficient(reason: str, confidence: Optional[float], cacheable: bool) -> dict:
        return {
            "outcome": "insufficient_evidence", "resolution_type": None, "summary": None,
            "scope_conditions": None, "confidence": confidence, "evidence_used": [], "positions": [],
            "insufficient_evidence_reason": reason, "_cacheable": cacheable,
        }

    def _not_a_conflict(reason: str, confidence: Optional[float], cacheable: bool) -> dict:
        # Reuses the same "insufficient_evidence_reason" dict key / SQLite
        # column as _insufficient — both are SQLite-only control signals with a
        # free-text reason; not_a_conflict didn't warrant its own column. The
        # prompt asks the model for a distinct "not_a_conflict_reason" field
        # (clearer for the model to fill), mapped into this shared slot here.
        return {
            "outcome": "not_a_conflict", "resolution_type": None, "summary": None,
            "scope_conditions": None, "confidence": confidence, "evidence_used": [], "positions": [],
            "insufficient_evidence_reason": reason, "_cacheable": cacheable,
        }

    if verdict is None:
        return _insufficient("LLM call failed after retries", None, cacheable=False)

    resolution_type = verdict.resolution_type
    summary = verdict.summary
    scope_conditions = verdict.scope_conditions
    confidence = verdict.confidence
    evidence_used = list(verdict.evidence_used)
    positions = [p.model_dump() for p in verdict.positions]
    insufficient_reason = verdict.insufficient_evidence_reason

    # Fix 2, step 0: checked BEFORE any "why do they oppose" branch below —
    # scope_difference especially is trivially satisfiable for almost any two
    # claims from different papers, so it must never substitute for asking
    # whether they oppose each other at all.
    if resolution_type == "not_a_conflict":
        return _not_a_conflict(
            verdict.not_a_conflict_reason or "model determined the claims do not make opposing assertions",
            confidence, cacheable=True,
        )

    # AMENDMENT 4: "unresolved" asserts scope was checked; it must not mean
    # "we could not look."
    if resolution_type == "unresolved":
        if not any_context:
            logger.warning(
                f"Overriding model's 'unresolved' to 'insufficient_evidence' for cluster {cluster}: "
                f"zero participants had scope context available, so scope was not checkable at all."
            )
            resolution_type = "insufficient_evidence"
            insufficient_reason = "model returned 'unresolved' with no scope context available for any participant"
        elif not all_context:
            if "partial_context" not in evidence_used:
                evidence_used.append("partial_context")
            confidence = (confidence or 0.0) * conf.partial_context_confidence_factor

    if resolution_type == "insufficient_evidence":
        return _insufficient(insufficient_reason or "model returned insufficient_evidence", confidence, cacheable=True)

    # §3.2/§8 invariant: scope_conditions present iff resolution_type == scope_difference.
    if resolution_type == "scope_difference" and not scope_conditions:
        logger.warning(
            f"Malformed verdict for cluster {cluster}: resolution_type=scope_difference but no "
            f"scope_conditions — overriding to insufficient_evidence rather than writing an invalid node."
        )
        return _insufficient("malformed verdict: scope_difference without scope_conditions", confidence, cacheable=False)
    if resolution_type != "scope_difference" and scope_conditions:
        scope_conditions = None  # strip rather than reject the whole verdict

    if not summary or not summary.strip():
        logger.warning(f"Malformed verdict for cluster {cluster}: empty summary — overriding to insufficient_evidence.")
        return _insufficient("malformed verdict: empty summary", confidence, cacheable=False)

    return {
        "outcome": "written", "resolution_type": resolution_type, "summary": summary,
        "scope_conditions": scope_conditions, "confidence": confidence, "evidence_used": evidence_used,
        "positions": positions, "insufficient_evidence_reason": None, "_cacheable": True,
    }


def _classify_clusters(clusters: List[List[str]], kg: KnowledgeGraph, store: CandidateStore,
                        client: ClassificationClient, conf: ClassificationConfig, desc_fields: Dict[str, dict],
                        scope_context: Dict[str, List[dict]], existing_records: Dict[str, ClusterRecord]) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    cluster_keys = [participants_hash(c) for c in clusters]
    # AMENDMENT 1: batched Neo4j-side verification, not trusting SQLite alone.
    exact_matches = _fetch_exact_hash_matches(kg, cluster_keys)

    to_process: List[tuple] = []
    skipped = 0
    for cluster, key in zip(clusters, cluster_keys):
        record = existing_records.get(key)
        sqlite_fresh = record is not None and record.pipeline_version == conf.pipeline_version
        if sqlite_fresh and key in exact_matches:
            skipped += 1
            continue
        to_process.append((cluster, key, record))

    stats = {"skipped": skipped, "processed": 0, "written": 0, "insufficient_evidence": 0,
              "not_a_conflict": 0, "by_resolution_type": {}, "merges": 0}

    if not to_process:
        return stats

    all_participant_ids = sorted({pid for cluster, _, _ in to_process for pid in cluster})
    overlap_map = _fetch_overlapping_contradictions(kg, all_participant_ids)

    # Clique decomposition (§4.2) means a single OLD Contradiction can now need
    # to become MULTIPLE new clusters — overlap_map is a snapshot from BEFORE
    # this run's writes, so two sibling cliques split from the same old node
    # would both see themselves as the sole owner of that old id if not tracked
    # here. Each old id may be claimed (reused OR deleted) by at most one
    # cluster per run; every other cluster that maps to an already-claimed id
    # is treated as having no prior overlap at all, so it correctly spawns a
    # fresh node instead of stomping on the first cluster's just-written edges.
    # Processed in a deterministic order (sorted by cluster_key) so which
    # cluster "wins" a shared old id is reproducible, not networkx-iteration-
    # order-dependent.
    claimed_contradiction_ids: set = set()

    for cluster, cluster_key, record in sorted(to_process, key=lambda t: t[1]):
        stats["processed"] += 1
        cache_key = _cluster_cache_key(cluster_key, conf.model, conf.classification_prompt_version, conf.pipeline_version)

        if record is not None and record.cache_key == cache_key:
            verdict_fields = {
                "outcome": record.outcome, "resolution_type": record.resolution_type,
                "summary": record.summary, "scope_conditions": record.scope_conditions,
                "confidence": record.confidence, "evidence_used": record.evidence_used,
                "positions": record.positions, "insufficient_evidence_reason": record.insufficient_evidence_reason,
                "_cacheable": True,
            }
        else:
            verdict_fields = _run_cluster_classification(cluster, client, conf, desc_fields, scope_context)

        cacheable = verdict_fields.pop("_cacheable", True)
        final_cache_key = cache_key if cacheable else None

        # Computed once per cluster, regardless of outcome, and immediately
        # claimed — see the comment above the loop for why (clique splits).
        overlapping: Dict[str, str] = {}
        for pid in cluster:
            for c_id, gen_at in overlap_map.get(pid, []):
                if c_id not in claimed_contradiction_ids:
                    overlapping[c_id] = gen_at
        claimed_contradiction_ids.update(overlapping.keys())

        if verdict_fields["outcome"] in ("insufficient_evidence", "not_a_conflict"):
            stats[verdict_fields["outcome"]] += 1
            if overlapping:
                # This outcome asserts nothing should be written — any node the
                # participants used to belong to (e.g. a mega-cluster that
                # decomposed and this sub-clique turned out not to be a
                # conflict, or a previously-written cluster that reclassified)
                # must not survive orphaned.
                _delete_contradictions(kg, list(overlapping.keys()))
                logger.info(
                    f"Deleted {len(overlapping)} stale Contradiction node(s) for cluster now "
                    f"classified {verdict_fields['outcome']}: {sorted(overlapping.keys())}"
                )
            store.upsert_cluster_record(ClusterRecord(
                cluster_key=cluster_key, participant_ids=cluster, contradiction_node_id=None,
                resolution_type=None, summary=None, scope_conditions=None,
                confidence=verdict_fields.get("confidence"), evidence_used=[], positions=[],
                outcome=verdict_fields["outcome"],
                insufficient_evidence_reason=verdict_fields.get("insufficient_evidence_reason"),
                cache_key=final_cache_key, pipeline_version=conf.pipeline_version,
            ))
            continue

        if not overlapping:
            keep_id, drop_ids = f"contradiction-{uuid4().hex}", []
        elif len(overlapping) == 1:
            keep_id, drop_ids = next(iter(overlapping)), []
        else:
            keep_id = min(overlapping, key=lambda cid: overlapping[cid])  # oldest generated_at survives
            drop_ids = [cid for cid in overlapping if cid != keep_id]
            stats["merges"] += 1
            logger.warning(
                f"Merging {len(drop_ids) + 1} previously-separate Contradictions into one "
                f"(cluster now has {len(cluster)} participants): kept={keep_id}, dropped={drop_ids}"
            )

        node_properties = {
            "summary": verdict_fields["summary"], "resolution_type": verdict_fields["resolution_type"],
            "scope_conditions": verdict_fields["scope_conditions"], "confidence": verdict_fields["confidence"],
            "participants_hash": cluster_key, "generated_by": f"{conf.model}|{conf.classification_prompt_version}",
            "generated_at": now, "pipeline_version": conf.pipeline_version,
            "evidence_used": verdict_fields["evidence_used"],
        }
        positions_map = {p["description_id"]: p["position"] for p in verdict_fields.get("positions", [])}

        with kg._driver.session(database=kg._database) as session:
            written_id = session.execute_write(
                _write_cluster_tx, cluster_ids=cluster, keep_id=keep_id, drop_ids=drop_ids,
                properties=node_properties, participant_positions=positions_map,
            )

        if written_id is None:
            # Amendment 2 fired (n < 2 after reconciliation). Structurally this
            # shouldn't happen for a freshly-computed cluster (always >= 2
            # participants by construction), but it's enforced unconditionally
            # per spec rather than assumed unreachable.
            stats["insufficient_evidence"] += 1
            store.upsert_cluster_record(ClusterRecord(
                cluster_key=cluster_key, participant_ids=cluster, contradiction_node_id=None,
                resolution_type=None, summary=None, scope_conditions=None, confidence=None,
                evidence_used=[], positions=[], outcome="insufficient_evidence",
                insufficient_evidence_reason="node fell below 2 participants after write (see logs)",
                cache_key=None, pipeline_version=conf.pipeline_version,
            ))
            continue

        stats["written"] += 1
        stats["by_resolution_type"][verdict_fields["resolution_type"]] = (
            stats["by_resolution_type"].get(verdict_fields["resolution_type"], 0) + 1
        )
        store.upsert_cluster_record(ClusterRecord(
            cluster_key=cluster_key, participant_ids=cluster, contradiction_node_id=written_id,
            resolution_type=verdict_fields["resolution_type"], summary=verdict_fields["summary"],
            scope_conditions=verdict_fields["scope_conditions"], confidence=verdict_fields["confidence"],
            evidence_used=verdict_fields["evidence_used"], positions=verdict_fields.get("positions", []),
            outcome="written", insufficient_evidence_reason=None,
            cache_key=final_cache_key, pipeline_version=conf.pipeline_version,
        ))

    return stats


# ---- top-level orchestrator ---------------------------------------------------

def run_classification(kg: KnowledgeGraph, store: CandidateStore, conf: ClassificationConfig) -> dict:
    """
    Stage 3 of the whole-KB conflict pass (docs/conflict_pipeline.md §4):
    supersession test (pairwise) -> clique-based clustering -> cluster
    classification, writing `supersedes` edges and `Contradiction` nodes/edges
    to Neo4j, with SQLite-only outcomes (insufficient_evidence, not_a_conflict)
    recorded but never written to the graph.
    """
    pairs = store.get_passed_pairs()
    if not pairs:
        logger.info("No gate-passed candidate pairs to classify.")
        return {"pairs_evaluated": 0, "clusters_total": 0, "skipped": 0, "processed": 0,
                "written": 0, "insufficient_evidence": 0, "not_a_conflict": 0, "by_resolution_type": {},
                "merges": 0, "supersedes_written": 0}

    source_ids = sorted({p.source_id_a for p in pairs} | {p.source_id_b for p in pairs})
    years = fetch_source_years(kg, source_ids)

    description_ids = sorted({p.description_id_a for p in pairs} | {p.description_id_b for p in pairs})
    desc_fields = fetch_description_fields(kg, description_ids)

    # One batched query for the whole run (better than "one per cluster" —
    # covers both the supersession-test context, amendment 3, and §4.3(a)'s
    # cluster-classification context with the same fetch).
    scope_context = fetch_scope_context(kg, source_ids)

    client = ClassificationClient(conf)

    supersession_updates, pairs_for_clustering = _process_supersession(
        pairs, years, desc_fields, scope_context, client, kg, conf
    )
    store.update_supersession_results(supersession_updates)

    clusters = _build_clusters(pairs_for_clustering, max_cluster_size=conf.max_cluster_size)
    existing_records = store.get_cluster_records()
    cluster_stats = _classify_clusters(
        clusters, kg, store, client, conf, desc_fields, scope_context, existing_records
    )

    stats = {
        "pairs_evaluated": len(pairs),
        "supersedes_written": sum(1 for u in supersession_updates if u.supersession_result == "yes"),
        "clusters_total": len(clusters),
        **cluster_stats,
    }
    logger.info(f"Classification complete: {stats}")
    return stats
