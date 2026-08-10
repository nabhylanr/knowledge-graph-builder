import os

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from src.utils.logger import get_logger
from typing import List

from src.agents.graph_extractor import GraphExtractor
from src.graph.graph_model import map_to_lc_graph, sanitize_graph
from src.config import LLMConf
from src.schema import ProcessedDocument

logger = get_logger(__name__)


class GraphMiner:
    """ Contains methods to mine graphs from a (list of) `ProcessedDocument`."""

    def __init__(self, conf: LLMConf):
        self.graph_extractor = GraphExtractor(conf=conf)

        if self.graph_extractor:
            logger.info(f"GraphMiner initialized.")


    def mine_graph_from_doc_chunks(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Mines a graph from a `ProcessedDocument` instance.
        """
        source_name = doc.filename or "unknown"
        source_format = (doc.metadata or {}).get("source_kind", source_name.rsplit(".", 1)[-1] if "." in source_name else "unknown")

        # Shared across ALL chunks of this document so the has_source cap is
        # enforced per-document, not per-chunk (otherwise an N-chunk document
        # could accumulate up to 3*N has_source edges instead of 3 total).
        has_source_state: dict = {}
        # Same per-document lifetime, so near-duplicate Topic strings merge across
        # chunks (a Topic from chunk 1 recognized when it reappears misspelled in
        # chunk 5) instead of only within a single chunk.
        topic_registry: dict = {}
        # Same per-document lifetime, so a Source `date`/`format` captured in one
        # chunk is still applied when a later chunk (with no date, or a different
        # one) re-emits and re-merges the same canonical Source node.
        source_meta_state: dict = {}

        # Phase 1 — extraction (the slow, LLM-bound part). Each chunk's extract is
        # independent (no cross-chunk state), so run them concurrently. Results are
        # collected in chunk order. Tune concurrency with EXTRACTOR_MAX_WORKERS
        # (lower it if the Ollama/GPU server can't keep up); 1 = fully sequential.
        max_workers = max(1, int(os.getenv("EXTRACTOR_MAX_WORKERS", "4")))

        # Chunks the producer marked as carrying nothing to extract (a transcript's
        # "you", "Thank you.") are not sent to the LLM at all. They are NOT dropped:
        # `store_chunks_for_doc` writes every chunk's text and embedding regardless
        # of whether a graph came back, so they keep their place in the NEXT chain
        # and stay retrievable — only the hallucinated Topic/Type/Description layer
        # on top of them is skipped. The extraction prompt cannot decline (it
        # requires 1-3 Topics per chunk), so this is the only place the decision
        # can be made.
        eligible = [c for c in doc.chunks if c.extraction_eligible]
        skipped = len(doc.chunks) - len(eligible)
        if skipped:
            logger.info(
                f"{source_name}: skipping extraction for {skipped}/{len(doc.chunks)} chunk(s) "
                f"marked extraction_eligible=false (stored and embedded, not extracted)."
            )
        if not eligible:
            logger.warning(f"{source_name}: no chunk is eligible for extraction — storing chunks only.")
            return doc

        # Progress counter — raw_only extraction logs nothing on success, so without
        # this a long run is silent until it finishes. Incremented under a lock since
        # workers run concurrently; logged every 10 chunks (and on the last one).
        total_chunks = len(eligible)
        progress = {"done": 0}
        progress_lock = Lock()

        def _extract(chunk):
            try:
                return self.graph_extractor.extract_graph(
                    text=chunk.text,
                    source_name=source_name,
                    source_format=source_format,
                )
            except Exception as e:
                logger.warning(f"Error while extracting graph: {e}")
                return None
            finally:
                with progress_lock:
                    progress["done"] += 1
                    if progress["done"] % 10 == 0 or progress["done"] == total_chunks:
                        logger.info(f"Extraction progress: {progress['done']}/{total_chunks} chunks")

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                raw_graphs = list(executor.map(_extract, eligible))
        else:
            raw_graphs = [_extract(chunk) for chunk in eligible]

        # Phase 2 — sanitize + map, SERIALLY in chunk order. This is deliberate:
        # sanitize mutates has_source_state and topic_registry, whose results are
        # order-dependent (first-3 has_source cap, first-spelling Topic dedup).
        # Parallelizing here would make the output non-deterministic, so it stays a
        # plain in-order loop — matching the previous sequential behaviour exactly.
        for chunk, graph in zip(eligible, raw_graphs):
            if graph is None:
                logger.warning(f"Skipping chunk — graph extraction returned None.")
                continue
            try:
                # Deterministically enforce the ontology (directions, has_source cap,
                # single Source, no self-loops) before mapping to the graph store.
                graph = sanitize_graph(
                    graph,
                    source_name=source_name,
                    has_source_state=has_source_state,
                    topic_registry=topic_registry,
                    source_meta_state=source_meta_state,
                )

                if graph is None:
                    logger.warning(f"Skipping chunk — no valid graph after sanitization.")
                    continue

                graph_doc = map_to_lc_graph(graph, source_content=chunk.text)

                chunk.nodes = graph_doc.nodes
                chunk.relationships = graph_doc.relationships

            except Exception as e:
                logger.warning(f"Error while mining graph: {e}")

        logger.info(f"Created a graph representation for {len(eligible)} chunks in {source_name}.")

        return doc


    def mine_graph_from_docs(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Mines graphs from a list of `ProcessedDocument` instances.
        """
        return [self.mine_graph_from_doc_chunks(doc) for doc in docs]
    