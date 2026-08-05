from pydantic import BaseModel
from typing import List, Optional

from src.prompts.cluster_classification import CLASSIFICATION_PROMPT_VERSION
from src.prompts.nli_gate import NLI_PROMPT_VERSION
from src.prompts.supersession import SUPERSESSION_PROMPT_VERSION

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
    min_similarity: Optional[float] = None  # no floor by default — see tail before cutting it

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

    # G5 — NLI verification (local Ollama; see src/conflict/nli_client.py)
    nli_model: str = "qwen2.5:3b"
    nli_prompt_version: str = NLI_PROMPT_VERSION
    nli_reject_confidence: float = 0.8
    nli_concurrency: int = 2  # keep modest — local Ollama, not a hosted API
    nli_timeout_seconds: int = 30  # a hung/slow local model must not block a worker forever — see nli_client.py

    # Bookkeeping — bump whenever G1-G5 logic or thresholds change, to force
    # re-evaluation of rows already gated under an older version.
    gate_version: str = "gate-v1"


class ClassificationConfig(BaseModel):
    """
    Tunables for stage 3 (classification & write-back) of the whole-KB
    conflict-detection pass. See docs/conflict_pipeline.md §4 for the decision
    rules and §3 for the output ontology this stage writes.

    Unlike gates (local dev model, fail-open, no retry), this stage uses the
    PRODUCTION model behind a real hosted API with real rate limits — retry
    conventions mirror src/agents/graph_extractor.py's GraphExtractor exactly
    (duplicated, not shared — see src/conflict/classification_client.py's
    module docstring for why).
    """
    model_type: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    api_key: Optional[str] = None

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

    # Bookkeeping — bump whenever decision rules, prompts, or thresholds change,
    # to force re-evaluation of pairs/clusters processed under an older version.
    pipeline_version: str = "classification-v1"
    sqlite_path: str = "conflict_candidates.db"
