import json
import re
import time
import traceback
from typing import List, Optional

from pydantic import BaseModel

from src.config import LLMConf
from src.conflict.config import ClassificationConfig
from src.factory.llm import fetch_llm
from src.prompts.cluster_classification import get_cluster_classification_prompt
from src.prompts.supersession import format_context_block, get_supersession_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)

# TECH DEBT (accepted): duplicated from src/agents/graph_extractor.py's
# GraphExtractor rather than factored into a shared helper. Extraction is
# working, load-bearing code with its own retry needs (small dev models,
# EXTRACTOR_RAW_ONLY, etc.); this stage's needs are similar but not identical
# (production model, no raw-only mode). Revisit if a third caller ever needs
# the same retry shape — until then, duplication is cheaper than coupling two
# independent pipelines through one shared utility.
_CONN_ERROR_MARKERS = (
    "connection refused", "server disconnected", "connection error",
    "max retries exceeded", "incomplete chunked read", "peer closed connection",
    "connection reset", "read timed out",
)

SUPERSESSION_ALLOWED_DECISIONS = {"yes", "no"}
SUPERSESSION_ALLOWED_BASIS = {"explicit_correction", "citing_contrast", "claimed_improvement", "retraction"}
CLUSTER_ALLOWED_RESOLUTION_TYPES = {"scope_difference", "known_controversy", "unresolved", "insufficient_evidence"}


class SupersessionVerdict(BaseModel):
    decision: str
    basis: Optional[str] = None
    reason: Optional[str] = None
    confidence: float


class ParticipantPosition(BaseModel):
    description_id: str
    position: str


class ClusterVerdict(BaseModel):
    resolution_type: str
    summary: Optional[str] = None
    scope_conditions: Optional[str] = None
    confidence: float
    evidence_used: List[str] = []
    positions: List[ParticipantPosition] = []
    insufficient_evidence_reason: Optional[str] = None


def _parse_json_block(content: str) -> Optional[dict]:
    """Same salvage technique as GraphExtractor's `_parse_graph_from_text` /
    NLIClient's `_parse_verdict_from_text` — extract the outermost JSON object
    from a raw completion that skipped or broke structured/tool calling."""
    if not content:
        return None
    content = re.sub(r"```[a-zA-Z]*", "", content).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(content[start:end + 1])
    except Exception:
        return None


class ClassificationClient:
    """
    Stage-3 classification LLM client — production model (Groq
    llama-3.3-70b-versatile by default), not the small dev model gates.py
    uses. Gates already absorbed the cheap-filtering work; this is the
    expensive-reasoning step, so it gets real retry behaviour instead of
    fail-open-on-first-error.
    """

    def __init__(self, conf: ClassificationConfig):
        self.conf = conf
        self.llm = fetch_llm(LLMConf(
            type=conf.model_type, model=conf.model, temperature=conf.temperature, api_key=conf.api_key,
        ))
        self.supersession_prompt = get_supersession_prompt()
        self.cluster_prompt = get_cluster_classification_prompt()

    def _invoke_with_retry(self, prompt_str: str, schema) -> Optional[BaseModel]:
        """Mirrors GraphExtractor.extract_graph's retry loop: structured output
        first, raw-JSON salvage on empty/failed structured output, retry with
        wait on rate-limit/connection errors, give up (return None) otherwise."""
        if self.llm is None:
            return None

        for attempt in range(1, self.conf.max_retries + 1):
            try:
                result = self.llm.with_structured_output(schema=schema).invoke(prompt_str)
                if result is not None:
                    return result

                fallback = self._raw_json_fallback(prompt_str, schema)
                if fallback is not None:
                    logger.info("Recovered classification verdict via raw-JSON fallback.")
                return fallback

            except Exception as e:
                error_str = str(e)
                error_low = error_str.lower()
                is_rate_limit = "429" in error_str or "rate_limit" in error_low
                is_permanent = "limit_exceeded" in error_low and "limit: 0" in error_str
                is_conn_error = any(m in error_low for m in _CONN_ERROR_MARKERS)

                if ((is_rate_limit and not is_permanent) or is_conn_error) and attempt < self.conf.max_retries:
                    wait = self.conf.retry_wait_seconds if is_rate_limit else self.conf.conn_retry_wait_seconds
                    reason = "Rate limit hit" if is_rate_limit else "Connection error"
                    logger.warning(f"{reason} (attempt {attempt}/{self.conf.max_retries}). Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if not is_rate_limit:
                    fallback = self._raw_json_fallback(prompt_str, schema)
                    if fallback is not None:
                        logger.info("Recovered classification verdict via raw-JSON fallback (after tool error).")
                        return fallback

                logger.error(f"Error during classification LLM call (attempt {attempt}): {e}\n{traceback.format_exc()}")
                return None

        return None

    def _raw_json_fallback(self, prompt_str: str, schema) -> Optional[BaseModel]:
        try:
            raw = self.llm.invoke(prompt_str)
            content = raw.content if hasattr(raw, "content") else str(raw)
            data = _parse_json_block(content)
            if data is None:
                return None
            return schema(**data)
        except Exception as fe:
            logger.warning(f"Raw-JSON fallback failed: {fe}")
            return None

    def classify_supersession(
        self,
        older_text: str, older_year: int, older_context_texts: List[str],
        newer_text: str, newer_year: int, newer_context_texts: List[str],
    ) -> Optional[SupersessionVerdict]:
        prompt_str = self.supersession_prompt.format(
            older_year=older_year, older_text=older_text,
            older_context_block=format_context_block(older_context_texts),
            newer_year=newer_year, newer_text=newer_text,
            newer_context_block=format_context_block(newer_context_texts),
        )
        verdict = self._invoke_with_retry(prompt_str, SupersessionVerdict)
        if verdict is None:
            return None
        if verdict.decision not in SUPERSESSION_ALLOWED_DECISIONS:
            logger.warning(f"Supersession verdict had invalid decision {verdict.decision!r} — treating as unparseable.")
            return None
        if verdict.decision == "yes" and verdict.basis not in SUPERSESSION_ALLOWED_BASIS:
            logger.warning(f"Supersession 'yes' verdict had invalid basis {verdict.basis!r} — treating as unparseable.")
            return None
        return verdict

    def classify_cluster(self, participants_block: str, n: int) -> Optional[ClusterVerdict]:
        prompt_str = self.cluster_prompt.format(n=n, participants_block=participants_block)
        verdict = self._invoke_with_retry(prompt_str, ClusterVerdict)
        if verdict is None:
            return None
        if verdict.resolution_type not in CLUSTER_ALLOWED_RESOLUTION_TYPES:
            logger.warning(f"Cluster verdict had invalid resolution_type {verdict.resolution_type!r} — treating as unparseable.")
            return None
        return verdict
