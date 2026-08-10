import os

from pydantic import BaseModel
from typing import List, Optional

from src.prompts.cluster_classification import CLASSIFICATION_PROMPT_VERSION
from src.prompts.nli_gate import NLI_PROMPT_VERSION
from src.prompts.supersession import SUPERSESSION_PROMPT_VERSION


def resolve_ollama_endpoint(explicit: Optional[str]) -> Optional[str]:
    """
    Shared endpoint-resolution order for every Ollama-backed stage of the
    conflict pass (gating, classification): a stage-specific env var first,
    then RE_MODEL_ENDPOINT (extraction's var — all stages normally point at
    the same remote host), then None (ChatOllama's own localhost:11434
    default in src/factory/llm.py). Both run_gates.py's build_gate_config()
    and run_classification.py's build_classification_config() call this, so
    the fallback behaves identically in both places rather than being
    reimplemented twice.
    """
    return explicit or os.getenv("RE_MODEL_ENDPOINT")

# Only assertion-bearing Types are worth comparing for conflicts — Background,
# Method, Dataset, etc. don't carry a claim that can be "wrong" the way a
# Result or a Decision can. See docs/conflict_ontology.md.
DEFAULT_ALLOWED_TYPES = ["Result", "Metrics Evaluation", "Conclusion", "Decision", "Progress Update"]

# G2 rejects a pair when either side's Topic is a category label, not a claim
# anchor — two papers both having a Result about "Performance" says nothing
# about them conflicting. Seed list; grow via config, not code, as false
# candidates surface. Deliberately no document-frequency heuristic yet — the
# corpus is too small for percentages to be stable (see gates.py).
DEFAULT_GENERIC_TOPIC_BLOCKLIST = [
    "method", "methods", "methodology", "performance", "system", "results",
    "result", "evaluation", "experiment", "experiments", "approach",
    "model", "framework", "analysis", "discussion", "introduction",
    "background", "overview", "summary", "conclusion", "future work",
    # Added after the gold-set blocking run: extraction leaked Type names and
    # bare metric words into topicName (e.g. 'Output', 'Business performance').
    # method/methods/results/result/background/discussion/analysis/conclusion/
    # performance were already covered above (matching is case-insensitive —
    # see gates.py's _normalize_topic); only the genuinely new ones are added.
    "findings", "output", "business performance", "financial performance",
    "operational performance", "data",
]

# Post-write relationship types — UPPERCASE, deliberately.
#
# langchain_neo4j's Neo4jGraph.add_graph_documents uppercases every
# relationship type on write (_get_rel_import_query -> apoc.merge.relationship
# with row.type = el.type.replace(" ", "_").upper(), see
# langchain_neo4j/graphs/neo4j_graph.py:667) but leaves node LABELS untouched.
# So any Cypher that reads from or writes directly to Neo4j — bypassing
# add_graph_documents, as this pass's raw MERGE/MATCH statements do — must use
# these uppercase forms to match what's actually stored, or every identity
# lookup silently matches nothing (see the classifier.py incident this
# constant exists to prevent from recurring: 7 lowercase occurrences across
# _write_supersedes/_write_cluster_tx and their read-side helpers, all wrong,
# all silent).
#
# Do NOT derive these by `.upper()`-ing a pre-write constant (e.g.
# graph_model.SUPERSEDES_RELATION) at the call site — the two layers must stay
# visibly, intentionally separate: pre-write (extraction prompt, sanitize_graph,
# _FIXED_RELATION_DIRS) is lowercase by the ontology's own design, post-write
# (this file's constants) is uppercase because that's what Neo4j actually holds
# once add_graph_documents has run. One being silently derived from the other
# is how this bug happens again.
SUPERSEDES_TYPE = "SUPERSEDES"
HAS_CONTRADICTION_TYPE = "HAS_CONTRADICTION"


class BlockingConfig(BaseModel):
    """
    Tunables for stage 1 (candidate generation / blocking) of the whole-KB
    conflict-detection pass. See docs/conflict_ontology.md for the 3-stage
    pipeline this is the first stage of.
    """
    # S2 (kNN) knobs
    k: int = 15
    # Floor set from the gold-set run (8 docs, 322 Descriptions): unfiltered
    # kNN gave 2371 S2 pairs, min 0.719 / avg 0.850 / max 1.000, most of them
    # topically-close but not opposed (e.g. 'Output' vs 'Units Completed Per
    # Unit Time' at 0.871). 0.85 cuts that down to ~1105.
    #
    # NOT 0.90: the Franke-1978/Levitt-2009 pair (Hawthorne case — the only
    # gold-set pair exercising the SUPERSEDES branch) sits at 0.879 similarity.
    # A 0.90 floor drops it silently and makes SUPERSEDES untestable. Do not
    # raise this above 0.879 without first re-checking that pair still clears it.
    min_similarity: Optional[float] = 0.85

    # Shared pair-generation rules
    allowed_types: List[str] = DEFAULT_ALLOWED_TYPES

    # Description embedding knobs
    prepend_topic_name: bool = False
    description_index_name: str = "description_vector"

    # Bookkeeping
    pipeline_version: str = "blocking-v1"
    sqlite_path: str = "conflict_candidates.db"


class GateConfig(BaseModel):
    """
    Tunables for stage 2 (gate filtering) of the whole-KB conflict-detection
    pass. Governing principle: FAIL OPEN — every gate errs toward KEEPING a
    pair when unsure. A false rejection is invisible (we never see the conflict
    we silently dropped); a false pass costs one LLM call downstream in stage 3.
    See docs/conflict_ontology.md and src/conflict/gates.py.
    """
    # G1 — degenerate text
    min_text_length: int = 40

    # G2 — generic topic
    generic_topic_blocklist: List[str] = DEFAULT_GENERIC_TOPIC_BLOCKLIST

    # G3 — intra-document near-duplicate (difflib.SequenceMatcher ratio)
    intra_doc_similarity_threshold: float = 0.9

    # G5 — NLI verification (Ollama; see src/conflict/nli_client.py)
    nli_model: str = "qwen3:4b"
    # Resolved by resolve_ollama_endpoint() above; unset -> localhost:11434.
    # Point at the remote Ollama host when the model lives there rather than on
    # this machine — otherwise every G5 call errors out and, fail-open, silently
    # passes every pair through ungated.
    nli_endpoint: Optional[str] = None
    nli_prompt_version: str = NLI_PROMPT_VERSION
    nli_reject_confidence: float = 0.8
    nli_concurrency: int = 2  # keep modest — one small Ollama box, not a hosted API
    # 120s, not 30s: a reasoning model (qwen3:4b) emits a thinking block
    # before answering, which regularly exceeds 30s — that was misclassified as
    # an infra ERROR_LABEL rather than a genuine model verdict. Still bounded
    # (see nli_client.py's _invoke_with_timeout) so a truly hung call can't
    # block a worker forever.
    nli_timeout_seconds: int = 120

    # Batch size for incremental SQLite flushes during G5 (see gates.py's
    # run_gates) — verdicts are written every N completions, not only once at
    # the end, so progress is visible and an interrupted run resumes instead
    # of restarting.
    gate_flush_batch_size: int = 25

    # Bookkeeping — bump whenever G1-G5 logic or thresholds change, to force
    # re-evaluation of rows already gated under an older version.
    gate_version: str = "gate-v1"


class ClassificationConfig(BaseModel):
    """
    Tunables for stage 3 (classification & write-back) of the whole-KB
    conflict-detection pass. See docs/conflict_pipeline.md §4 for the decision
    rules and §3 for the output ontology this stage writes.

    Unlike gates (fail-open, no retry), this is the expensive-reasoning step and
    a dropped verdict is a real loss — so it retries. Retry conventions mirror
    src/agents/graph_extractor.py's GraphExtractor exactly (duplicated, not
    shared — see src/conflict/classification_client.py's module docstring for
    why), including the 429/rate-limit branch: dead weight against Ollama, but
    kept so the two loops stay diffable.

    `model_type` stays configurable, but Ollama is the only provider the stack
    actually has — ModelType in src/config.py has no GROQ member since Groq was
    removed on 2026-08-03, so anything else fails at fetch_llm() time.
    """
    model_type: str = "ollama"
    model: str = "qwen3:4b"
    temperature: float = 0.0
    api_key: Optional[str] = None
    # Same role as GateConfig.nli_endpoint — unset means localhost:11434.
    endpoint: Optional[str] = None

    # Retry conventions, mirroring GraphExtractor's module constants
    max_retries: int = 5
    retry_wait_seconds: int = 65        # rate limit (429)
    conn_retry_wait_seconds: int = 30   # transient connection error

    # §3.1 — the SAME constant blocking uses for candidate-type eligibility,
    # reused here for supersedes endpoint validation, not a second copy.
    allowed_types: List[str] = DEFAULT_ALLOWED_TYPES

    supersession_prompt_version: str = SUPERSESSION_PROMPT_VERSION
    classification_prompt_version: str = CLASSIFICATION_PROMPT_VERSION

    # §4.3(c) "unresolved" partial-context confidence penalty — multiplicative,
    # applied when some but not all participants had scope context available.
    partial_context_confidence_factor: float = 0.7

    # §4.2 clique clustering safety net: a maximal clique found within a
    # connected component of passed pairs is truncated to this many
    # participants (sorted ids, first N kept) before being sent to the LLM —
    # decomposition into maximal cliques bounds a NON-clique component, but a
    # genuinely dense clique (every pair independently passed the gate) could
    # still exceed a size the classification prompt can reason about sensibly.
    max_cluster_size: int = 5

    # Bookkeeping — bump whenever decision rules, prompts, or thresholds change,
    # to force re-evaluation of pairs/clusters processed under an older version.
    # v2: clique-based clustering (was connected components) + not_a_conflict
    # outcome — bumped so every existing cluster record is re-processed, not
    # just ones whose participant set (and therefore cluster_key) changed shape.
    pipeline_version: str = "classification-v2"
    sqlite_path: str = "conflict_candidates.db"
