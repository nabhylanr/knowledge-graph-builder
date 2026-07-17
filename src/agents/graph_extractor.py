import json
import re
import time
import traceback

from src.utils.logger import get_logger
from typing import Optional

from src.factory.llm import fetch_llm
from src.config import LLMConf
from src.graph.graph_model import Ontology, _Graph
from src.prompts.graph_extractor import get_graph_extractor_prompt


logger = get_logger(__name__)

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 65  # wait 65s between retries on rate limit


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

    def __init__(self, conf: LLMConf, ontology: Optional[Ontology]=None):
        self.conf = conf
        self.llm = fetch_llm(conf)
        self.prompt = get_graph_extractor_prompt()


    def extract_graph(self, text: str, source_name: str = "unknown", source_format: str = "unknown") -> _Graph:
        """
        Extracts a graph from a text. Retries on rate limit errors.
        """

        if self.llm is None:
            return None

        prompt_str = self.prompt.format(
            input_text=text,
            source_name=source_name,
            source_format=source_format
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
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
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()

                is_permanent = "limit: 0" in error_str or "limit_exceeded" in error_str.lower() and "limit: 0" in error_str

                if is_rate_limit and not is_permanent and attempt < MAX_RETRIES:
                    logger.warning(f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). Waiting {RETRY_WAIT_SECONDS}s...")
                    time.sleep(RETRY_WAIT_SECONDS)
                    continue

                # Non-rate-limit failure (e.g. Groq `tool_use_failed`): the model
                # often emits perfectly valid JSON that only the tool-calling wrapper
                # rejected. Salvage it via a plain completion before giving up.
                if not is_rate_limit:
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