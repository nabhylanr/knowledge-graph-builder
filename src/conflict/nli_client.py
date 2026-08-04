import concurrent.futures
import json
import re
from typing import Callable, Optional, TypeVar

from pydantic import BaseModel

from src.config import LLMConf, ModelType
from src.factory.llm import fetch_llm
from src.prompts.nli_gate import get_nli_gate_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Labels the model itself may legitimately return.
ALLOWED_LABELS = {"contradiction", "entailment", "neutral", "unclear"}

# Reserved label for an INFRASTRUCTURE failure (timeout, connection error,
# unparseable/empty response) — NOT a model judgement. Both "unclear" (the
# model genuinely couldn't decide) and "error" (we never got a usable verdict)
# pass the gate identically (fail open), but are recorded distinguishably so
# the audit can tell "model was unsure" apart from "Ollama hiccuped" instead of
# both collapsing into one uninterpretable "unclear" bucket.
ERROR_LABEL = "error"


class NLIVerdict(BaseModel):
    label: str
    confidence: float
    rationale: str


def _parse_verdict_from_text(content: str) -> Optional[NLIVerdict]:
    """Best-effort JSON salvage from a raw completion — same technique as
    GraphExtractor's `_parse_graph_from_text` for the same class of small model
    that often fails structured/tool calling but still emits valid JSON as text."""
    if not content:
        return None
    content = re.sub(r"```[a-zA-Z]*", "", content).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(content[start:end + 1])
        return NLIVerdict(**data)
    except Exception:
        return None


class NLIClient:
    """
    G5 (NLI verification) LLM client — Ollama, same client/config pattern
    as GraphExtractor (src/agents/graph_extractor.py), no new model or
    dependency. Deliberately NO retry loop: per docs/conflict_ontology.md's
    fail-open principle, every error path (timeout, parse failure, connection
    error) resolves to a single `ERROR_LABEL` verdict immediately rather than
    being retried — an infrastructure hiccup must never silently drop a
    candidate, but it also shouldn't stall the batch waiting on a flaky local
    model when "pass and record it" is already the correct fallback.

    Timeout note: `LLMConf.timeout` (-> `client_kwargs={"timeout": ...}` in
    `fetch_llm`) does NOT reliably bound wall-clock time for a streaming Ollama
    response — httpx's per-chunk read timeout can keep getting reset by a slow
    trickle of tokens, so a hung/slow model can still block past the configured
    timeout (verified empirically: it did not fire). `_invoke_with_timeout`
    below is the actual enforcement — a bounded wait on a background thread —
    independent of whatever the HTTP client does internally. Trade-off: on a
    timeout, the underlying call is NOT cancelled, just abandoned; the request
    thread keeps running until Ollama itself finishes or errors out. Acceptable
    for a short-lived batch gate run with modest concurrency — not something to
    do at high volume without also capping total background threads.
    """

    def __init__(self, model: str, prompt_version: str, timeout_seconds: int = 30,
                 endpoint: Optional[str] = None):
        self.model = model
        self.prompt_version = prompt_version
        self.timeout_seconds = timeout_seconds
        self.llm = fetch_llm(LLMConf(type=ModelType.OLLAMA, model=model, temperature=0.0,
                                     timeout=timeout_seconds, endpoint=endpoint))
        self.prompt = get_nli_gate_prompt()

    def _invoke_with_timeout(self, fn: Callable[[], T]) -> T:
        # NOT `with ThreadPoolExecutor() as executor:` — that context manager's
        # __exit__ calls shutdown(wait=True), which blocks until the abandoned
        # background call finishes, silently defeating the timeout from this
        # method's own caller's perspective (verified empirically: with `with`,
        # a 5s-configured timeout still took 26s wall-clock because __exit__
        # waited for the real, slow response anyway). shutdown(wait=False) here
        # returns as soon as `.result()` raises, regardless of the future's
        # eventual outcome.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            return executor.submit(fn).result(timeout=self.timeout_seconds)
        finally:
            executor.shutdown(wait=False)

    def get_verdict(self, text_a: str, text_b: str) -> NLIVerdict:
        if self.llm is None:
            return NLIVerdict(label=ERROR_LABEL, confidence=0.0, rationale="LLM not configured")

        prompt_str = self.prompt.format(text_a=text_a, text_b=text_b)

        try:
            verdict = self._invoke_with_timeout(
                lambda: self.llm.with_structured_output(schema=NLIVerdict).invoke(prompt_str)
            )
        except concurrent.futures.TimeoutError:
            logger.warning(f"NLI call timed out after {self.timeout_seconds}s — recording as error (infra), not model 'unclear'.")
            return NLIVerdict(label=ERROR_LABEL, confidence=0.0, rationale=f"timeout after {self.timeout_seconds}s")
        except Exception as e:
            logger.warning(f"NLI call failed ({e}) — recording as error (infra), not model 'unclear'.")
            return NLIVerdict(label=ERROR_LABEL, confidence=0.0, rationale=f"error: {e}")

        if verdict is not None and verdict.label in ALLOWED_LABELS:
            return verdict

        # Structured/tool calling failed or returned an out-of-vocabulary label
        # (not a timeout/exception) — salvage via one raw completion, same
        # fallback GraphExtractor uses for the same class of small model. Also
        # individually timeout-bounded; worst case here is ~2x timeout_seconds.
        try:
            raw = self._invoke_with_timeout(lambda: self.llm.invoke(prompt_str))
            fallback = _parse_verdict_from_text(raw.content if hasattr(raw, "content") else str(raw))
            if fallback is not None and fallback.label in ALLOWED_LABELS:
                return fallback
            logger.warning("NLI verdict unparseable or invalid label — recording as error (infra), not model 'unclear'.")
            return NLIVerdict(label=ERROR_LABEL, confidence=0.0, rationale="unparseable verdict")
        except concurrent.futures.TimeoutError:
            logger.warning(f"NLI fallback call timed out after {self.timeout_seconds}s — recording as error (infra), not model 'unclear'.")
            return NLIVerdict(label=ERROR_LABEL, confidence=0.0, rationale=f"fallback timeout after {self.timeout_seconds}s")
        except Exception as e:
            logger.warning(f"NLI fallback call failed ({e}) — recording as error (infra), not model 'unclear'.")
            return NLIVerdict(label=ERROR_LABEL, confidence=0.0, rationale=f"fallback error: {e}")
