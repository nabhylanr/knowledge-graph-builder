"""
The storage layer: what actually reaches Neo4j for one chunk and one document.

The theme is that `add_embeddings` and `add_graph_documents` both compile to
Cypher this repo does not control, and both of them silently drop or clobber
properties in ways that only show up in the graph weeks later. Everything here
pins the workarounds in place.

No Neo4j: the Cypher is asserted as text and the driver calls are recorded.
"""

import pytest

from langchain_neo4j.graphs.graph_document import Node

from src.graph.knowledge_graph import KnowledgeGraph
from src.schema import Chunk, ProcessedDocument


class FakeVectorStore:
    """Stands in for Neo4jVector. `add_embeddings` returns the ids it wrote, the
    same as the real one (md5 of the text)."""

    def __init__(self):
        self.calls = []
        self.indexes_created = 0

    def add_embeddings(self, texts, embeddings, metadatas):
        self.calls.append({"texts": texts, "embeddings": embeddings, "metadatas": metadatas})
        return [f"md5({t})" for t in texts]

    def create_new_index(self):
        self.indexes_created += 1


class RecordingGraph(KnowledgeGraph):
    """A KnowledgeGraph with every Neo4j round-trip replaced by a list."""

    def __init__(self):  # deliberately does not call super(): no connection
        self.vector_store = FakeVectorStore()
        self.created_at_writes = []
        self.graph_documents = []
        self.mentions = []
        self.next_chains = []
        self.document_nodes = []
        self.contradiction_cleanups = 0
        self.source_metadata = []

    def set_chunk_created_at_if_absent(self, chunk_id, created_at):
        self.created_at_writes.append((chunk_id, created_at))

    def add_graph_documents(self, graph_documents, include_source, baseEntityLabel):
        self.graph_documents.extend(graph_documents)

    def create_mentions_relationships(self, node_id, chunk_id, filename, document_version):
        self.mentions.append((node_id, chunk_id))

    def create_next_relationships(self, filename, doc_version):
        self.next_chains.append(filename)

    def create_document_node(self, doc):
        self.document_nodes.append(dict(doc.metadata or {}))

    def cleanup_singleton_contradictions(self):
        self.contradiction_cleanups += 1

    def write_source_metadata(self, source_id, props):
        self.source_metadata.append((source_id, props))


class FakeTx:
    def __init__(self):
        self.runs = []

    def run(self, query, **params):
        self.runs.append((query, params))


def doc_with(*chunks, metadata=None):
    return ProcessedDocument(
        filename="paper.jsonl",
        metadata=metadata if metadata is not None else {"source_kind": "pdf"},
        chunks=list(chunks),
    )


def chunk(chunk_id=0, text="Some text.", nodes=None):
    return Chunk(chunk_id=chunk_id, text=text, embedding=[0.1, 0.2], nodes=nodes)


class TestChunkCreatedAt:
    def test_created_at_never_goes_through_chunk_metadata(self):
        """`add_embeddings` compiles to `MERGE (c:Chunk {id}) ... SET c += row.metadata`
        with no ON CREATE/ON MATCH split, so a metadata entry would be
        overwritten with "now" on every re-write — and re-writes are routine once
        a resumed run re-touches chunks already in the graph."""
        kg = RecordingGraph()
        kg.store_single_chunk(doc_with(), chunk())
        assert "created_at" not in kg.vector_store.calls[0]["metadatas"][0]

    def test_the_guarded_write_uses_the_vector_stores_own_id(self):
        """It must address the same node `add_embeddings` just MERGEd — the md5 of
        the text, not Chunk.chunk_id. Taken from the return value so it cannot
        drift from the library's convention."""
        kg = RecordingGraph()
        kg.store_single_chunk(doc_with(), chunk(chunk_id=7, text="Some text."))
        assert [cid for cid, _ in kg.created_at_writes] == ["md5(Some text.)"]

    def test_the_cypher_only_sets_an_absent_timestamp(self):
        """The `IS NULL` guard is the whole feature: created_at records when a
        chunk FIRST entered the graph, not when it was last touched."""
        tx = FakeTx()
        KnowledgeGraph._set_chunk_created_at_if_absent(tx, "abc", "2026-08-12T00:00:00+00:00")
        query, params = tx.runs[0]
        assert "c.created_at IS NULL" in query
        assert params == {"chunk_id": "abc", "created_at": "2026-08-12T00:00:00+00:00"}

    def test_a_failed_chunk_write_does_not_stamp_a_timestamp(self):
        kg = RecordingGraph()
        kg.vector_store.add_embeddings = lambda **kw: (_ for _ in ()).throw(RuntimeError("neo4j down"))
        assert kg.store_single_chunk(doc_with(), chunk()) is None
        assert kg.created_at_writes == []


class TestDocumentCreatedAt:
    def test_it_is_a_query_parameter_not_a_metadata_key(self):
        """`doc.metadata` is the document's own dict, shared with everything else
        that reads it during a build — a per-call timestamp does not belong in
        it."""
        tx = FakeTx()
        KnowledgeGraph._create_document_node(tx, doc_with())
        query, params = tx.runs[0]
        assert "created_at: $created_at" in query
        assert params["created_at"]
        assert "created_at" not in params["metadata"]

    def test_every_document_gets_one(self):
        tx = FakeTx()
        KnowledgeGraph._create_document_node(tx, ProcessedDocument(filename="a.jsonl", chunks=[]))
        KnowledgeGraph._create_document_node(tx, ProcessedDocument(filename="b.jsonl", chunks=[]))
        assert all(params["created_at"] for _, params in tx.runs)


class TestChunkMetadata:
    def test_chunk_keys_do_not_leak_into_the_document_metadata(self):
        """This used to alias `metadata = doc.metadata` and mutate it, so
        `_create_document_node`'s `SET d += $metadata` wrote the LAST chunk's
        chunk_id/chunk_size onto the Document node."""
        doc = doc_with(metadata={"source_kind": "pdf"})
        RecordingGraph().store_single_chunk(doc, chunk(chunk_id=41))
        assert doc.metadata == {"source_kind": "pdf"}

    def test_the_chunk_still_carries_its_own_identity(self):
        kg = RecordingGraph()
        kg.store_single_chunk(doc_with(), chunk(chunk_id=41))
        written = kg.vector_store.calls[0]["metadatas"][0]
        assert written["chunk_id"] == 41
        assert written["filename"] == "paper.jsonl"
        assert written["source_kind"] == "pdf"


class TestSourceAccumulation:
    def _source_chunk(self, chunk_id, **props):
        return chunk(chunk_id=chunk_id, text=f"text {chunk_id}", nodes=[
            Node(id="Paper Jsonl", type="Source", properties={"name": "Paper Jsonl", **props}),
        ])

    def test_the_last_chunk_carrying_a_source_wins(self):
        """sanitize_graph's source_meta_state only ever gains keys across chunks,
        so the last snapshot is the fullest one."""
        kg = RecordingGraph()
        doc = doc_with()
        first = kg.store_single_chunk(doc, self._source_chunk(0))
        second = kg.store_single_chunk(doc, self._source_chunk(1, year="2021"))
        assert first == ("Paper Jsonl", {"name": "Paper Jsonl"})
        assert second == ("Paper Jsonl", {"name": "Paper Jsonl", "year": "2021"})

    def test_a_chunk_with_no_graph_reports_no_source(self):
        assert RecordingGraph().store_single_chunk(doc_with(), chunk()) is None

    def test_finalize_writes_the_accumulated_metadata(self):
        kg = RecordingGraph()
        kg.finalize_document(
            doc_with(),
            accumulated_source_props={"name": "Paper Jsonl", "year": "2021", "date_raw": "2021-08"},
            source_node_id="Paper Jsonl",
        )
        assert kg.source_metadata == [("Paper Jsonl", {"year": "2021", "date_raw": "2021-08"})]

    def test_a_producer_date_outranks_the_models_reading(self):
        kg = RecordingGraph()
        kg.finalize_document(
            doc_with(metadata={"source_kind": "transcript", "date": "2026-03-11"}),
            accumulated_source_props={"year": "1999", "date_raw": "sometime in 1999"},
            source_node_id="Meeting Jsonl",
        )
        assert kg.source_metadata == [("Meeting Jsonl", {"year": "2026", "date_raw": "2026-03-11"})]


class TestBatchPathIsUnchanged:
    """`store_chunks_for_doc` now just drives store_single_chunk + finalize, and
    has to keep behaving exactly as it did for any caller still using it."""

    def test_every_chunk_is_stored_and_the_document_finalized_once(self):
        kg = RecordingGraph()
        kg.store_chunks_for_doc(doc_with(chunk(0), chunk(1, "b"), chunk(2, "c")))
        assert len(kg.vector_store.calls) == 3
        assert kg.next_chains == ["paper.jsonl"]
        assert len(kg.document_nodes) == 1
        assert kg.contradiction_cleanups == 1
        assert kg.vector_store.indexes_created == 1

    def test_the_source_write_still_happens_at_the_end(self):
        kg = RecordingGraph()
        kg.store_chunks_for_doc(doc_with(
            chunk(0),
            chunk(1, "b", nodes=[Node(id="Paper Jsonl", type="Source", properties={"year": "2021"})]),
        ))
        assert kg.source_metadata == [("Paper Jsonl", {"year": "2021"})]

    def test_mentions_are_created_per_chunk(self):
        kg = RecordingGraph()
        kg.store_chunks_for_doc(doc_with(
            chunk(0, nodes=[Node(id="Chunking", type="Topic", properties={})]),
            chunk(1, "b", nodes=[Node(id="Retrieval", type="Topic", properties={})]),
        ))
        assert kg.mentions == [("Chunking", 0), ("Retrieval", 1)]
