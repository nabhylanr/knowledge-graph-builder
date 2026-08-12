import os

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from src.utils.logger import get_logger
from typing import Iterator, List, Optional, Tuple

from src.agents.graph_extractor import GraphExtractor
from src.graph.graph_model import (
    _canonical_id,
    classify_expected_domain,
    map_to_lc_graph,
    sanitize_graph,
)
from src.ingestion.build_ledger import content_digest
from src.config import LLMConf
from src.schema import Chunk, ProcessedDocument

logger = get_logger(__name__)


class GraphMiner:
    """ Contains methods to mine graphs from a (list of) `ProcessedDocument`."""

    def __init__(self, conf: LLMConf):
        self.graph_extractor = GraphExtractor(conf=conf)

        if self.graph_extractor:
            logger.info(f"GraphMiner initialized.")


    @staticmethod
    def _doc_context(doc: ProcessedDocument) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        `(source_name, source_format, doc_type, expected_domain)` for a document.

        `doc_type` (the free-text folder name) is the SOFT prompt hint;
        `expected_domain` (derived from the controlled `source_kind` enum) is the
        hard one sanitize_graph deletes on. See `classify_expected_domain` for
        why they are not the same input.
        """
        source_name = doc.filename or "unknown"
        metadata = doc.metadata or {}
        source_format = metadata.get(
            "source_kind",
            source_name.rsplit(".", 1)[-1] if "." in source_name else "unknown"
        )
        return source_name, source_format, metadata.get("doc_type"), classify_expected_domain(source_format)


    def _extract_in_order(
        self,
        chunks: List[Chunk],
        source_name: str,
        source_format: str,
        doc_type: Optional[str],
    ) -> Iterator[Tuple[Chunk, object]]:
        """
        Yield `(chunk, raw_graph)` in CHUNK ORDER while extracting concurrently.

        Order matters even though extraction itself is order-independent: the
        caller's `sanitize_graph` state (`has_source_state`, `topic_registry`,
        `source_meta_state`) is mutated per chunk and its results are
        order-dependent — which chunk wins the first-3 has_source cap, and which
        spelling of a Topic becomes canonical. Yielding as each extraction
        completes would make both non-deterministic, so results are consumed in
        order while later chunks keep extracting in the background. The cost is
        only latency on a chunk whose predecessor is still running; throughput is
        unchanged, and the output is identical to a fully serial run.

        Tune concurrency with EXTRACTOR_MAX_WORKERS (lower it if the Ollama/GPU
        server can't keep up); 1 = fully sequential. `raw_graph` is None for a
        chunk whose extraction failed — never raises.
        """
        max_workers = max(1, int(os.getenv("EXTRACTOR_MAX_WORKERS", "4")))

        # Progress counter — raw_only extraction logs nothing on success, so without
        # this a long run is silent until it finishes. Incremented under a lock since
        # workers run concurrently; logged every 10 chunks (and on the last one).
        total_chunks = len(chunks)
        progress = {"done": 0}
        progress_lock = Lock()

        def _extract(chunk: Chunk):
            try:
                return self.graph_extractor.extract_graph(
                    text=chunk.text,
                    source_name=source_name,
                    source_format=source_format,
                    doc_type=doc_type,
                )
            except Exception as e:
                logger.warning(f"Error while extracting graph: {e}")
                return None
            finally:
                with progress_lock:
                    progress["done"] += 1
                    if progress["done"] % 10 == 0 or progress["done"] == total_chunks:
                        logger.info(f"Extraction progress: {progress['done']}/{total_chunks} chunks")

        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [executor.submit(_extract, chunk) for chunk in chunks]
            for chunk, future in zip(chunks, futures):
                yield chunk, future.result()
        finally:
            # Not a `with` block on purpose: its __exit__ waits for every queued
            # extraction to finish, so abandoning this generator (the consumer
            # raised, or the process is dying) would hang for as long as the
            # remaining chunks take — minutes each on a local model.
            executor.shutdown(wait=False, cancel_futures=True)


    @staticmethod
    def _log_extraction_skips(doc: ProcessedDocument, source_name: str) -> None:
        """
        Report the chunks that will not get an LLM call.

        Chunks the producer marked as carrying nothing to extract (a transcript's
        "you", "Thank you.") are not sent to the LLM at all. They are NOT dropped:
        every chunk's text and embedding is written regardless of whether a graph
        came back, so they keep their place in the NEXT chain and stay
        retrievable — only the hallucinated Topic/Type/Description layer on top of
        them is skipped. The extraction prompt cannot decline (it requires 1-3
        Topics per chunk), so this is the only place the decision can be made.
        """
        skipped = sum(1 for c in doc.chunks if not c.extraction_eligible)
        if skipped:
            logger.info(
                f"{source_name}: skipping extraction for {skipped}/{len(doc.chunks)} chunk(s) "
                f"marked extraction_eligible=false (stored and embedded, not extracted)."
            )


    def mine_graph_from_doc_chunks(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Mines a graph from a `ProcessedDocument` instance, attaching each chunk's
        nodes/relationships to the chunk in memory.

        The batch path: nothing reaches Neo4j until a caller stores the document
        afterwards, so a crash costs everything mined so far. Prefer
        `mine_and_store_doc_chunks`, which interleaves the two.
        """
        source_name, source_format, doc_type, expected_domain = self._doc_context(doc)

        # Shared across ALL chunks of this document so the has_source cap is
        # enforced per-document, not per-chunk (otherwise an N-chunk document
        # could accumulate up to 3*N has_source edges instead of 3 total), so
        # near-duplicate Topic strings merge across chunks, and so a Source
        # `date`/`format` captured in one chunk survives onto later ones.
        has_source_state: dict = {}
        topic_registry: dict = {}
        source_meta_state: dict = {}

        self._log_extraction_skips(doc, source_name)
        eligible = [c for c in doc.chunks if c.extraction_eligible]
        if not eligible:
            logger.warning(f"{source_name}: no chunk is eligible for extraction — storing chunks only.")
            return doc

        for chunk, raw_graph in self._extract_in_order(eligible, source_name, source_format, doc_type):
            try:
                self._attach_graph_to_chunk(
                    chunk, raw_graph, source_name, expected_domain,
                    has_source_state, topic_registry, source_meta_state,
                )
            except Exception as e:
                logger.warning(f"Error while mining graph: {e}")

        logger.info(f"Created a graph representation for {len(eligible)} chunks in {source_name}.")

        return doc


    @staticmethod
    def _attach_graph_to_chunk(
        chunk: Chunk,
        raw_graph,
        source_name: str,
        expected_domain: Optional[str],
        has_source_state: dict,
        topic_registry: dict,
        source_meta_state: dict,
    ) -> bool:
        """
        Sanitize one chunk's raw graph and attach it to the chunk. True when the
        chunk ended up with a graph.

        Deterministically enforces the ontology (directions, has_source cap,
        single Source, no self-loops, document domain) before mapping to the
        graph store. MUST be called in chunk order — it mutates the three state
        dicts, whose results are order-dependent.
        """
        if raw_graph is None:
            logger.warning("Skipping chunk — graph extraction returned None.")
            return False

        graph = sanitize_graph(
            raw_graph,
            source_name=source_name,
            has_source_state=has_source_state,
            topic_registry=topic_registry,
            source_meta_state=source_meta_state,
            expected_domain=expected_domain,
        )
        if graph is None:
            logger.warning("Skipping chunk — no valid graph after sanitization.")
            return False

        graph_doc = map_to_lc_graph(graph, source_content=chunk.text)
        chunk.nodes = graph_doc.nodes
        chunk.relationships = graph_doc.relationships
        return True


    def mine_and_store_doc_chunks(
        self,
        doc: ProcessedDocument,
        knowledge_graph,
        checkpoint=None,
    ) -> ProcessedDocument:
        """
        Extract and store one document CHUNK BY CHUNK, checkpointing each chunk
        once it is in Neo4j.

        The point of interleaving: extraction is the slow part, and until this
        existed nothing reached Neo4j until every chunk of every pending document
        had been extracted. A crash at chunk 80 of 129 lost all 80, and a crash
        on document 3 of 5 lost documents 1 and 2 as well. Neither half of the
        fix works alone — a checkpoint that says "chunk 79 done" is worthless if
        chunk 79's graph only ever existed in memory.

        Storage was already idempotent (Chunk MERGEs by content hash, entities by
        canonical id), so what resuming actually saves is the LLM calls and the
        wall clock — plus the non-determinism of a small local model re-extracting
        the same chunk and legitimately returning a different Topic set.

        `checkpoint`: a `ChunkCheckpoint`, or None to extract-and-store without
        any resume support.
        """
        source_name, source_format, doc_type, expected_domain = self._doc_context(doc)

        # Content fingerprint, so a checkpoint written for an older revision of
        # this document is discarded rather than resumed onto different text.
        digest = content_digest(doc)

        done_ids = checkpoint.done_chunk_ids(doc.filename, doc.document_version, digest) if checkpoint else set()
        # Resuming restores the cross-chunk sanitizer state, not just the chunk
        # list: starting these from empty would let a resumed document exceed
        # MAX_HAS_SOURCE document-wide and re-canonicalize a Topic under a second
        # spelling, i.e. produce a graph a single uninterrupted run never would.
        resumed = checkpoint.sanitizer_state(doc.filename, doc.document_version, digest) if checkpoint else {}
        has_source_state: dict = resumed.get("has_source", {})
        topic_registry: dict = resumed.get("topics", {})
        source_meta_state: dict = resumed.get("source_meta", {})

        def _snapshot() -> dict:
            return {
                "has_source": has_source_state,
                "topics": topic_registry,
                "source_meta": source_meta_state,
            }

        def _mark_done(chunk: Chunk) -> None:
            if checkpoint:
                checkpoint.mark_done(
                    doc.filename, doc.document_version, chunk.chunk_id,
                    digest=digest, state=_snapshot(),
                )

        self._log_extraction_skips(doc, source_name)

        # A checkpointed chunk is already IN Neo4j — that is what "done" means
        # here — so it is neither re-extracted nor re-stored. Chunks that are
        # merely ineligible still need storing (text + embedding + their place in
        # the NEXT chain), they just never see the LLM.
        pending = [c for c in doc.chunks if c.chunk_id not in done_ids]
        if len(pending) < len(doc.chunks):
            logger.info(
                f"{source_name}: resuming — {len(doc.chunks) - len(pending)} chunk(s) already "
                f"in the graph, {len(pending)} to go."
            )

        source_node: Optional[Tuple[str, dict]] = None

        for chunk in pending:
            if chunk.extraction_eligible:
                continue
            knowledge_graph.store_single_chunk(doc, chunk)
            _mark_done(chunk)

        to_extract = [c for c in pending if c.extraction_eligible]
        for chunk, raw_graph in self._extract_in_order(to_extract, source_name, source_format, doc_type):
            try:
                extracted = self._attach_graph_to_chunk(
                    chunk, raw_graph, source_name, expected_domain,
                    has_source_state, topic_registry, source_meta_state,
                )
                stored = knowledge_graph.store_single_chunk(doc, chunk)
                if stored:
                    source_node = stored
            except Exception as e:
                logger.warning(f"Error while mining chunk {chunk.chunk_id} of {source_name}: {e}")
                continue

            # ONLY now, after the write. Marking a chunk done at extraction time
            # would let a crash in between silently lose it: resume trusts the
            # checkpoint and would never extract it again.
            #
            # A chunk whose extraction FAILED is deliberately left unmarked even
            # though its text and embedding are now stored (re-storing is
            # idempotent). Extraction failures are usually transient — a server
            # blip, a model that returned unparseable JSON — so if this run later
            # crashes, a resume gets one more attempt at them instead of
            # inheriting the gap permanently.
            if extracted:
                _mark_done(chunk)

        if source_node is None:
            # No chunk in this pass carried a Source node — either none of them
            # did, or the run resumed with every chunk already written. The
            # accumulated date/format still has to reach Neo4j, and
            # `source_meta_state` holds exactly the keys finalize_document wants.
            canon_source = _canonical_id(source_name)
            source_meta = source_meta_state.get(canon_source)
            if source_meta:
                source_node = (canon_source, source_meta)

        knowledge_graph.finalize_document(
            doc,
            accumulated_source_props=source_node[1] if source_node else None,
            source_node_id=source_node[0] if source_node else None,
        )

        if checkpoint:
            # Redundant with the BuildLedger entry the caller is about to write.
            checkpoint.clear_document(doc.filename, doc.document_version)

        logger.info(f"Stored {len(pending)} chunk(s) for {source_name}.")

        return doc


    def mine_graph_from_docs(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Mines graphs from a list of `ProcessedDocument` instances.
        """
        return [self.mine_graph_from_doc_chunks(doc) for doc in docs]
