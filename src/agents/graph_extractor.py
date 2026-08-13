import json
import os
import re
import time
import traceback

from src.utils.logger import get_logger
from typing import Optional

from src.factory.llm import fetch_llm
from src.config import LLMConf
from src.graph.graph_model import _Graph
from src.prompts.graph_extractor import get_graph_extractor_prompt


logger = get_logger(__name__)

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 65  # wait 65s between retries on rate limit
CONN_RETRY_WAIT_SECONDS = 30  # wait 30s between retries on a transient connection error

# Substrings that mark a transient server-connectivity failure (remote Ollama
# restarted / briefly down / network blip) — worth retrying rather than dropping
# the chunk. Without retry, a mid-run server restart permanently loses every
# in-flight/queued chunk even though the server recovers seconds later (observed:
# one ~15-min blip silently dropped ~50 of 106 chunks).
_CONN_ERROR_MARKERS = (
    "connection refused",
    "server disconnected",
    "connection error",
    "max retries exceeded",
    "incomplete chunked read",
    "peer closed connection",
    "connection reset",
    "read timed out",
)


def _parse_graph_from_text(content: str) -> Optional[_Graph]:
    """
    Best-effort parse of a model's raw text into a `_Graph`.

    Smaller models (e.g. llama-3.1-8b) often fail tool/function calling and make
    `with_structured_output` return an empty graph, yet emit perfectly valid JSON
    as plain text. This salvages that output by extracting the outermost JSON
    object and validating it against the schema.
    """
    if not content:
        return None
    # Strip markdown code fences if present.
    content = re.sub(r"```[a-zA-Z]*", "", content).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(content[start:end + 1])
    except Exception:
        return None
    try:
        return _Graph(
            nodes=data.get("nodes", []),
            relationships=data.get("relationships", []),
        )
    except Exception:
        return None


class GraphExtractor:
    """ Agent able to extract informations in a graph representation format from a given text.
    """

    def __init__(self, conf: LLMConf, corpus_domain: Optional[str] = None):
        self.conf = conf
        self.llm = fetch_llm(conf)
        # NOTE: the ontology (allowed node/relationship types and their directions)
        # is NOT configurable per-instance — it's fixed inside the prompt template
        # (src/prompts/graph_extractor.py) and re-enforced deterministically by
        # `sanitize_graph` (src/graph/graph_model.py). There used to be an
        # `ontology: Optional[Ontology]` parameter here that was threaded in from
        # `Configuration` but silently discarded; it was removed since it never
        # did anything.
        #
        # `corpus_domain` is different from that removed param: it actually
        # changes the prompt (R19 — see get_graph_extractor_prompt's v8.8 note),
        # restricting the Type vocabulary shown to the model when a whole build
        # is known to be paper-only or meeting-only. None (default) renders the
        # full, unrestricted vocabulary — unchanged behaviour for a mixed or
        # unclassified build.
        self.corpus_domain = corpus_domain
        self.prompt = get_graph_extractor_prompt(corpus_domain=corpus_domain)


    def extract_graph(
        self,
        text: str,
        source_name: str = "unknown",
        source_format: str = "unknown",
        doc_type: Optional[str] = None,
    ) -> Optional[_Graph]:
        """
        Extracts a graph from a text. Retries on rate limit errors.

        `doc_type` is the folder-derived classification of the document (set by
        `ChunksIngestor` from the chunk file's parent folder, e.g. "gold_b1",
        "meeting"). It reaches the prompt as a SOFT prior on which Type
        vocabulary to use — it does not remove vocabulary, so it only lowers the
        odds of a paper chunk being tagged with a meeting Type. The hard
        enforcement is `sanitize_graph(expected_domain=...)`, which keys off
        `source_kind` rather than this free-text folder name.
        """

        if self.llm is None:
            return None

        prompt_str = self.prompt.format(
            input_text=text,
            source_name=source_name,
            source_format=source_format,
            # PromptTemplate.format() requires every declared variable, and a
            # None would render literally as "None" — the prompt's fallback
            # branch keys off the word "unclassified" instead.
            doc_type=doc_type or "unclassified",
        )

        # qwen3 "soft switch": appending /no_think suppresses its reasoning block to
        # cut latency. Opt-in via env so default behaviour is unchanged. NOTE: the
        # qwen3-vl variant honours this only partially.
        if os.getenv("EXTRACTOR_NO_THINK") == "1":
            prompt_str = f"{prompt_str}\n\n/no_think"

        # Models like qwen3-vl:4b reliably fail tool/function calling —
        # `with_structured_output` returns an empty graph — yet emit valid JSON as
        # plain text. For those, every chunk wastes a full structured-output call
        # before falling back to a raw completion anyway (~half the latency is pure
        # waste). EXTRACTOR_RAW_ONLY skips straight to the raw path, so each chunk
        # makes one LLM call instead of two. Leave it unset for models whose
        # tool-calling actually works (e.g. OpenAI), which lose nothing from the
        # structured path.
        raw_only = os.getenv("EXTRACTOR_RAW_ONLY") == "1"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if raw_only:
                    # Single completion; parse may return None/empty, which the
                    # caller handles the same as any other failed extraction.
                    return _parse_graph_from_text(self._raw_completion(prompt_str))

                graph: _Graph = self.llm.with_structured_output(
                    schema=_Graph
                ).invoke(input=prompt_str)

                if graph is not None and graph.nodes:
                    return graph

                # Tool/function calling returned an empty graph — common with smaller
                # models that still emit valid JSON as plain text. Salvage that output.
                fallback = _parse_graph_from_text(self._raw_completion(prompt_str))
                if fallback is not None and fallback.nodes:
                    logger.info("Recovered graph via raw-JSON fallback.")
                    return fallback

                return graph

            except Exception as e:
                error_str = str(e)
                error_low = error_str.lower()
                is_rate_limit = "429" in error_str or "rate_limit" in error_low

                is_permanent = "limit_exceeded" in error_low and "limit: 0" in error_str

                # Transient connectivity failure (remote server restarted / blip):
                # retry with a wait so the chunk survives a short outage instead of
                # being dropped the instant the server refuses a connection.
                is_conn_error = any(m in error_low for m in _CONN_ERROR_MARKERS)

                if ((is_rate_limit and not is_permanent) or is_conn_error) and attempt < MAX_RETRIES:
                    wait = RETRY_WAIT_SECONDS if is_rate_limit else CONN_RETRY_WAIT_SECONDS
                    reason = "Rate limit hit" if is_rate_limit else "Connection error"
                    logger.warning(f"{reason} (attempt {attempt}/{MAX_RETRIES}). Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # Non-rate-limit failure (e.g. a `tool_use_failed`): the model
                # often emits perfectly valid JSON that only the tool-calling wrapper
                # rejected. Salvage it via a plain completion before giving up.
                # Skipped in raw_only mode — the raw completion is what just failed,
                # so re-running it would only waste another call.
                if not is_rate_limit and not raw_only:
                    try:
                        fallback = _parse_graph_from_text(self._raw_completion(prompt_str))
                        if fallback is not None and fallback.nodes:
                            logger.info("Recovered graph via raw-JSON fallback (after tool error).")
                            return fallback
                    except Exception as fe:
                        logger.warning(f"Raw-JSON fallback failed: {fe}")

                logger.error(f"Error while extracting graph (attempt {attempt}): {e}\n{traceback.format_exc()}")
                return None


    def _raw_completion(self, prompt_str: str) -> str:
        """Invoke the LLM without structured output and return its text content."""
        raw = self.llm.invoke(prompt_str)
        return raw.content if hasattr(raw, "content") else str(raw)