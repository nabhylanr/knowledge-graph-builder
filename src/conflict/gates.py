import difflib
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.conflict.candidate_store import CandidateStore, GateInputRow, GateResult
from src.conflict.config import GateConfig
from src.conflict.nli_client import ERROR_LABEL, NLIClient
from src.graph.knowledge_graph import KnowledgeGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)

# G4 (hedging / factuality) is deliberately NOT implemented in v1.
# `Description.text` is LLM-written prose summarising a source claim, not the
# author's original sentence — hedging cues ("may", "suggests", "preliminary")
# survive extraction unreliably, so judging claim certainty at this layer would
# be noise. Claim certainty is better judged by the stage-3 classification LLM,
# which reads fuller context than a two-sentence Description. Revisit only if
# G5 rejections turn out to correlate with hedged language once real data is in.


def _fetch_texts(kg: KnowledgeGraph, description_ids: List[str]) -> Dict[str, str]:
    """
    One batched query for every Description text this gate run needs — not
    N+1. (At corpus sizes far beyond today's, chunk the `IN $ids` list; not
    needed yet.)
    """
    if not description_ids:
        return {}
    rows = kg.query(
        "MATCH (d:Description) WHERE d.id IN $ids RETURN d.id AS id, d.text AS text",
        params={"ids": description_ids},
    )
    return {r["id"]: r["text"] or "" for r in rows}


def _is_degenerate(text: str, min_len: int) -> bool:
    return not text or not text.strip() or len(text.strip()) < min_len


def _normalize_topic(topic: str) -> str:
    return " ".join((topic or "").lower().split())


def _cache_key(id_a: str, id_b: str, text_a: str, text_b: str, model: str, prompt_version: str) -> str:
    h = hashlib.sha1()
    for part in (id_a, id_b, text_a, text_b, model, prompt_version):
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


@dataclass
class PendingResult:
    gate_status: str
    gate_reason: Optional[str]
    nli_label: Optional[str]
    nli_confidence: Optional[float]
    nli_cache_key: Optional[str]


def _evaluate_pair(
    row: GateInputRow, text_a: str, text_b: str, nli_client: NLIClient, conf: GateConfig
) -> PendingResult:
    """G1 -> G2 -> G3 -> G5, short-circuiting on first rejection. Fail open
    throughout: every gate below errs toward 'passed' when unsure."""

    # G1 — degenerate text
    if _is_degenerate(text_a, conf.min_text_length) or _is_degenerate(text_b, conf.min_text_length):
        return PendingResult("rejected", "degenerate_text", None, None, None)

    # G2 — generic topic (category label, not a claim anchor)
    blocklist = {_normalize_topic(t) for t in conf.generic_topic_blocklist}
    if _normalize_topic(row.topic_a) in blocklist or _normalize_topic(row.topic_b) in blocklist:
        return PendingResult("rejected", "generic_topic", None, None, None)

    # G3 — intra-document near-duplicate (extraction artifact, not a conflict).
    # Only applies within one document — cross-document near-duplicate text is
    # exactly what G5 should be allowed to call a contradiction or entailment.
    if row.source_id_a == row.source_id_b:
        similarity = difflib.SequenceMatcher(None, text_a, text_b).ratio()
        if similarity > conf.intra_doc_similarity_threshold:
            return PendingResult("rejected", "intra_doc_duplicate", None, None, None)

    # G4 — skipped, see module docstring above.

    # G5 — NLI verification, with cache.
    cache_key = _cache_key(row.description_id_a, row.description_id_b, text_a, text_b,
                            conf.nli_model, conf.nli_prompt_version)
    if row.nli_cache_key == cache_key and row.nli_label is not None:
        label, confidence = row.nli_label, row.nli_confidence
        result_cache_key = cache_key
    else:
        verdict = nli_client.get_verdict(text_a, text_b)
        label, confidence = verdict.label, verdict.confidence
        # Do NOT cache an infrastructure error under this key — caching it would
        # permanently "poison" the pair (identical texts -> identical cache key
        # forever) even after a transient Ollama hiccup resolves. A genuine
        # "unclear" verdict IS cached (the model actually judged it); only
        # ERROR_LABEL is left uncached so the next run retries the LLM call.
        result_cache_key = None if label == ERROR_LABEL else cache_key

    if label == "contradiction":
        return PendingResult("passed", None, label, confidence, result_cache_key)
    if label in ("unclear", ERROR_LABEL):
        return PendingResult("passed", None, label, confidence, result_cache_key)
    if label in ("entailment", "neutral") and confidence is not None and confidence >= conf.nli_reject_confidence:
        reason = "nli_entailment" if label == "entailment" else "nli_neutral"
        return PendingResult("rejected", reason, label, confidence, result_cache_key)
    return PendingResult("passed", None, label, confidence, result_cache_key)


def run_gates(kg: KnowledgeGraph, store: CandidateStore, conf: GateConfig) -> dict:
    """
    Stage 2 of the whole-KB conflict pass (docs/conflict_ontology.md): cheap
    rejection (G1-G3) then NLI verification (G5) of every candidate pair not
    yet gated under the current `gate_version`. Updates rows in place — never
    deletes. No writes to Neo4j; Description texts are read, nothing else.
    """
    rows = store.rows_needing_gate(conf.gate_version)
    if not rows:
        logger.info("No candidate pairs need gating.")
        return {"evaluated": 0, "passed": 0, "rejected": 0, "by_reason": {}}

    ids = sorted({d for r in rows for d in (r.description_id_a, r.description_id_b)})
    texts = _fetch_texts(kg, ids)

    nli_client = NLIClient(model=conf.nli_model, prompt_version=conf.nli_prompt_version,
                           timeout_seconds=conf.nli_timeout_seconds, endpoint=conf.nli_endpoint)

    def _work(row: GateInputRow):
        text_a = texts.get(row.description_id_a, "")
        text_b = texts.get(row.description_id_b, "")
        return row.pair_key, _evaluate_pair(row, text_a, text_b, nli_client, conf)

    now = datetime.now(timezone.utc).isoformat()
    results: List[GateResult] = []
    with ThreadPoolExecutor(max_workers=max(1, conf.nli_concurrency)) as executor:
        for pair_key, pending in executor.map(_work, rows):
            results.append(GateResult(
                pair_key=pair_key,
                gate_status=pending.gate_status,
                gate_reason=pending.gate_reason,
                gate_version=conf.gate_version,
                gate_checked_at=now,
                nli_label=pending.nli_label,
                nli_confidence=pending.nli_confidence,
                nli_cache_key=pending.nli_cache_key,
            ))

    store.update_gate_results(results)

    stats = {
        "evaluated": len(results),
        "passed": sum(1 for r in results if r.gate_status == "passed"),
        "rejected": sum(1 for r in results if r.gate_status == "rejected"),
        "by_reason": {},
    }
    for r in results:
        if r.gate_reason:
            stats["by_reason"][r.gate_reason] = stats["by_reason"].get(r.gate_reason, 0) + 1
    logger.info(f"Gate filtering complete: {stats}")
    return stats
