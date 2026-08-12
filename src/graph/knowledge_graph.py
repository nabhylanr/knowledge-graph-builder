import networkx as nx

from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from neo4j import ManagedTransaction
from typing import List, Optional, Set, Tuple

from src.config import KnowledgeGraphConfig
from src.graph.graph_ds import (
    build_update_query,
    compute_centralities,
    detect_leiden_communities,
    detect_louvain_communities,
    update_modularity
)
from src.schema import ProcessedDocument
from src.utils.logger import get_logger


logger = get_logger(__name__)

BASE_ENTITY_LABEL = "__Entity__"

# Entity labels the extraction ontology produces (src/prompts/graph_extractor.py)
# that never get `created_at` from `add_graph_documents`: see
# `_write_source_metadata`'s docstring — `onMatchProperties` is hardcoded empty,
# so only the chunk whose write FIRST creates a node sets its properties, and
# the extraction prompt never asks the model for a timestamp in the first
# place. Document and Chunk already get one elsewhere (`_create_document_node`,
# `_set_chunk_created_at_if_absent`), so they're not in this list.
#
# kg-agent's temporal gate (node_trust.py) is the actual reader of this field
# (brief R7); that repo was not available to confirm which labels beyond Topic
# it checks — extend this list if it turns out to read Agent/Type/Source/
# Description too.
ENTITY_LABELS_NEEDING_CREATED_AT = ["Topic"]


class KnowledgeGraph(Neo4jGraph):
    """
        Class used to represent a Knowledge Base under graph representation,
        using `neo4j` as the backend for querying operations.

        The ontology (allowed node/relationship types and their directions) is
        NOT configurable per-instance — it's fixed inside the extraction prompt
        (`src/prompts/graph_extractor.py`) and re-enforced deterministically by
        `sanitize_graph` (`src/graph/graph_model.py`).
    """

    def __init__(
            self,
            conf: KnowledgeGraphConfig,
            embeddings_model: Embeddings,
            sanitize = False,
            refresh_schema = True,
            enhanced_schema = False
        ):
        if conf.uri is not None:
            self.url = conf.uri
        else:
            self.url = f"{conf.db_schema}://{conf.host_name}:{conf.port}"
        self.username = conf.user
        self.password = conf.password
        self.database = conf.database
        self.timeout = conf.timeout
        self.index_name = conf.index_name

        self.embeddings = embeddings_model

        self._labels_ = None
        self._number_of_entities_ = None
        self._number_of_labels_ = None
        self._number_of_relationships_ = None
        self._number_of_docs = None
        self._relationships_ = None
        self._leiden_modularity = None
        self._number_of_leiden_communities = None
        self._louvain_modularity = None
        self._number_of_louvain_communities = None

        try:
            self.vector_store = Neo4jVector(
                embedding=self.embeddings,
                url=self.url,
                username=self.username,
                database=self.database,
                password=self.password,
                index_name=self.index_name,
                node_label="Chunk",
                embedding_node_property="embedding",
                text_node_property="text",
            )
        except Exception as e:
            logger.warning(f"Error connecting to Neo4jVector: {e}")

        super().__init__(
            url=self.url,
            username=self.username,
            password=self.password,
            database=self.database,
            timeout=self.timeout,
            sanitize=sanitize,
            refresh_schema=refresh_schema,
            enhanced_schema=enhanced_schema
        )


    @property
    def labels(self) -> List[str]:
        """
        Returns a list of labels in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "CALL db.labels() YIELD label RETURN COLLECT(label) AS labels"
            result = session.run(query)
            self._labels = result.single()["labels"]
        return self._labels


    @property
    def relationships(self) -> List[str]:
        """
        Returns a list of relationships in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "CALL db.relationshipTypes() YIELD relationshipType RETURN COLLECT(relationshipType) AS relationship_types"
            result = session.run(query)
            self._relationships_ = result.single()["relationship_types"]
        return self._relationships_


    @property
    def number_of_nodes(self) -> int:
        """
        Returns the total number of nodes in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "MATCH (n) RETURN COUNT(n) AS nodes"
            result = session.run(query)
            self._number_of_entities = result.single()["nodes"]
        return self._number_of_entities


    @property
    def number_of_labels(self) -> int:
        """
        Returns the number of labels in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "CALL db.labels() YIELD label RETURN COUNT(label) AS num_labels"
            result = session.run(query)
            self._number_of_labels = result.single()["num_labels"]
        return self._number_of_labels


    @property
    def number_of_relationships(self) -> int:
        """
        Returns the total number of relationships in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "MATCH ()-[r]-() RETURN COUNT(r) AS num_relationships"
            result = session.run(query)
            self._number_of_relationships = result.single()["num_relationships"]
        return self._number_of_relationships


    @property
    def number_of_docs(self) -> int:
        """
        Returns the current number of documents collected in the Knowledge Graph
        """
        with self._driver.session(database=self._database) as session:
            query = "MATCH (n: Document) RETURN COUNT(n) AS num_docs"
            result = session.run(query)
            self._number_of_docs = result.single()["num_docs"]
        return self._number_of_docs


    def document_filenames(self) -> Set[str]:
        """
        Every `filename` a Document node currently carries — the graph's own
        answer to "what is already built", which the build ledger is reconciled
        against so a wiped database does not leave the ledger lying.
        """
        query = "MATCH (d: Document) WHERE d.filename IS NOT NULL RETURN DISTINCT d.filename AS filename"
        with self._driver.session(database=self._database) as session:
            return {record["filename"] for record in session.run(query)}


    @property
    def leiden_modularity(self) -> float:
        query = """MATCH (m:GraphMetric WHERE m.name = 'leiden_modularity') RETURN m.value AS mod"""
        with self._driver.session(database=self._database) as session:
            try:
                result = session.run(query)
                self._leiden_modularity = result.single().value()
                return self._leiden_modularity
            except Exception as e:
                logger.warning("Leiden Modularity has not been computed")


    @property
    def louvain_modularity(self) -> float:
        query = """MATCH (m:GraphMetric WHERE m.name = 'louvain_modularity') RETURN m.value AS mod"""
        with self._driver.session(database=self._database) as session:
            try:
                result = session.run(query)
                self._louvain_modularity = result.single().value()
                return self._louvain_modularity
            except Exception as e:
                logger.warning("Louvain Modularity has not been computed")


    @property
    def number_of_louvain_communities(self) -> int:
        query = """
            MATCH (n)
            WHERE n.community_louvain IS NOT NULL
            RETURN count(DISTINCT n.community_louvain) AS num_communities
        """
        with self._driver.session(database=self._database) as session:
            try:
                result = session.run(query)
                self._number_of_louvain_communities = result.single()["num_communities"]
                return self._number_of_louvain_communities
            except Exception as e:
                logger.warning("Louvain communities have not been detected yet")


    @property
    def number_of_leiden_communities(self) -> int:
        query = """
            MATCH (n)
            WHERE n.community_leiden IS NOT NULL
            RETURN count(DISTINCT n.community_leiden) AS num_communities
        """
        with self._driver.session(database=self._database) as session:
            try:
                result = session.run(query)
                self._number_of_leiden_communities = result.single()["num_communities"]
                return self._number_of_leiden_communities
            except Exception as e:
                logger.warning("Leiden communities have not been detected yet")


    @staticmethod
    def _create_document_node(tx: ManagedTransaction, doc: ProcessedDocument):
        # `created_at` is a query parameter in the CREATE map, deliberately NOT
        # folded into `doc.metadata`: that dict is the document's own, shared
        # with everything else that reads it during a build, and a per-call
        # timestamp has no business being written into it. CREATE (not MERGE)
        # means this runs exactly once per document, so setting it directly is
        # safe — no ON CREATE guard needed, unlike Chunk (see
        # `set_chunk_created_at_if_absent`).
        query = """
            CREATE (d:Document {
                filename: $filename,
                document_version: $document_version,
                created_at: $created_at
            })
            SET d += $metadata
        """
        try:
            tx.run(
                query,
                filename=doc.filename,
                document_version=doc.document_version,
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata=doc.metadata or {},
            )
            logger.info(f"Document node created for file: {doc.filename}")
        except Exception as e:
            logger.warning(f"Error creating Document node for file: {doc.filename}: {e}")


    @staticmethod
    def _create_part_of_relationships(tx: ManagedTransaction, filename: str, document_version: int):
        query = """
            MATCH (d:Document {filename: $filename, document_version: $document_version})
            MATCH (c:Chunk {filename: $filename, document_version: $document_version})
            MERGE (c)-[:PART_OF]->(d)
        """
        try:
            tx.run(query, filename=filename, document_version=document_version)
            logger.info(f"PART_OF relationships created for Document {filename} version {document_version}")
        except Exception as e:
            logger.warning(f"Error creating PART_OF relationships for Document {filename}: {e}")


    @staticmethod
    def _create_next_relationships(
        tx: ManagedTransaction,
        filename: str,
        document_version: int
        ):
        query = """
            MATCH (c1:Chunk {filename: $filename, document_version: $document_version})
            WITH c1
            MATCH (c2:Chunk {filename: $filename, document_version: $document_version, chunk_id: c1.chunk_id + 1})
            MERGE (c1)-[:NEXT]->(c2)
        """
        try:
            tx.run(query, filename=filename, document_version=document_version)
        except Exception as e:
            logger.warning(f"Error creating NEXT relationships for chunks in Document {filename}: {e}")


    @staticmethod
    def _create_precedes_relationships(tx: ManagedTransaction, series: str):
        """
        Chains every Document node sharing the same `series` metadata property
        (e.g. a recurring meeting name) into a PRECEDES sequence ordered by
        their `date` property (an ISO 8601 string, e.g. "2026-07-20", sorts
        correctly lexicographically). Mirrors `_create_next_relationships`
        (Chunk-level) but at the Document level — without it there is no way
        to ask "what came before this week's Decision" across meetings.
        Documents missing `date` are excluded rather than guessed at.
        """
        query = """
            MATCH (d:Document {series: $series})
            WHERE d.date IS NOT NULL
            WITH d ORDER BY d.date ASC
            WITH collect(d) AS docs
            UNWIND range(0, size(docs) - 2) AS i
            WITH docs[i] AS d1, docs[i + 1] AS d2
            MERGE (d1)-[:PRECEDES]->(d2)
        """
        try:
            tx.run(query, series=series)
            logger.info(f"PRECEDES relationships created for Document series '{series}'")
        except Exception as e:
            logger.warning(f"Error creating PRECEDES relationships for series '{series}': {e}")


    @staticmethod
    def _cleanup_singleton_contradictions(tx: ManagedTransaction):
        """
        Deletes Contradiction nodes left with fewer than 2 distinct Description
        participants (MIN_CONTRADICTION_PARTICIPANTS in src/graph/graph_model.py).

        `sanitize_graph` can't enforce this: it sees one chunk at a time, so a
        Contradiction with 1 edge in chunk N might gain its 2nd participant in a
        later chunk — dropping it eagerly would lose the already-persisted first
        edge. Running once, after ALL chunks are stored, checks the real graph
        instead of a partial per-chunk view (and also catches the rarer case of
        two chunks reusing the same Contradiction id).

        Safe to run unconditionally and repeatedly: a genuine Contradiction keeps
        >=2 edges once its document is fully ingested, so only leftovers go.
        """
        query = """
            MATCH (c:Contradiction)
            WHERE COUNT { (c)<-[:HAS_CONTRADICTION]-(:Description) } < 2
            DETACH DELETE c
        """
        try:
            tx.run(query)
            logger.info("Removed Contradiction nodes with fewer than 2 Description participants")
        except Exception as e:
            logger.warning(f"Error cleaning up singleton Contradiction nodes: {e}")
            


    @staticmethod
    def _create_mentions_relationships(
        tx: ManagedTransaction,
        node_id: str,
        chunk_id: int,
        filename: str,
        document_version: int
        ):
        query = """
            MATCH (c:Chunk {chunk_id: $chunk_id, filename: $filename, document_version: $document_version})
            MATCH (e {id: $node_id}) WHERE NOT e:Chunk AND NOT e:Document
            MERGE (c)-[:MENTIONS]->(e)
        """
        try:
            tx.run(
                query,
                node_id=node_id,
                chunk_id=chunk_id,
                filename=filename,
                document_version=document_version
            )
        except Exception as e:
            logger.warning(f"Error creating MENTIONS relationships for {node_id}: {e}")


    @staticmethod
    def _set_chunk_created_at_if_absent(tx: ManagedTransaction, chunk_id: str, created_at: str):
        """
        `WHERE c.created_at IS NULL` is the whole point: this must record when a
        chunk FIRST entered the graph, and a chunk is re-written routinely — a
        resumed run re-touches chunks that were already in Neo4j before the
        crash, and `--rebuild` rewrites everything.
        """
        query = """
            MATCH (c:Chunk {id: $chunk_id})
            WHERE c.created_at IS NULL
            SET c.created_at = $created_at
        """
        try:
            tx.run(query, chunk_id=chunk_id, created_at=created_at)
        except Exception as e:
            logger.warning(f"Error setting created_at for chunk {chunk_id}: {e}")


    @staticmethod
    def _set_entity_created_at_if_absent(tx: ManagedTransaction, label: str, created_at: str):
        """
        Same fix as `_set_chunk_created_at_if_absent`, for an extracted entity
        label instead of Chunk: `add_graph_documents` (baseEntityLabel=False)
        compiles to `apoc.merge.node([type], {id}, row.properties, {})` with
        onMatchProperties hardcoded empty, so only the chunk that FIRST creates
        a node writes its properties — and the extraction prompt never asks the
        model for a timestamp anyway. `WHERE created_at IS NULL` makes this
        first-write-wins: whichever document's ingest introduces a given node
        (by id) stamps it once; a later document's write to the same node
        (MERGE by id, e.g. a Type shared across documents) leaves it untouched.

        `label` is interpolated into the query, not parameterized — Cypher has
        no way to parameterize a label. Safe here because it only ever comes
        from `ENTITY_LABELS_NEEDING_CREATED_AT`, a fixed internal list, never
        user input.
        """
        query = f"""
            MATCH (n:{label})
            WHERE n.created_at IS NULL
            SET n.created_at = $created_at
        """
        try:
            tx.run(query, created_at=created_at)
        except Exception as e:
            logger.warning(f"Error setting created_at for {label} nodes: {e}")


    @staticmethod
    def _write_source_metadata(tx: ManagedTransaction, source_id: str, props: dict):
        """
        `SET s += $props` only touches the keys present in `props` — never clears
        a property that isn't included, so a re-ingest that captures fewer keys
        than a previous run can't erase a previously-written good value.

        No existence check needed: if no Source node with this id exists, the
        MATCH matches zero rows and SET is a no-op — that's the "skip silently"
        behaviour, for free, from Cypher semantics.
        """
        query = """
            MATCH (s:Source {id: $source_id})
            SET s += $props
        """
        try:
            tx.run(query, source_id=source_id, props=props)
            logger.info(f"Wrote Source metadata {list(props.keys())} for '{source_id}'")
        except Exception as e:
            logger.warning(f"Error writing Source metadata for '{source_id}': {e}")


    def index_exists(self) -> bool:
        dimensions, index_ent_type = self.vector_store.retrieve_existing_index()
        if not dimensions:
            return False
        else:
            return True


    def create_index(self) -> bool:
        try:
            self.vector_store.create_new_index()
            return True
        except:
            return False


    def create_document_node(self, doc: ProcessedDocument):
        """
        Creates a Document node in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_document_node,
                doc
            )
            session.execute_write(
                self._create_part_of_relationships,
                doc.filename,
                doc.document_version
            )
            logger.info(f"Document node created for file: {doc.filename}")


    def create_next_relationships(self, filename: str, doc_version: int):
        """
        Creates NEXT relationships between Chunk Nodes from a Document.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_next_relationships,
                filename,
                doc_version
            )
            logger.info(f"NEXT relationships created for Document {filename} version {doc_version}")


    def create_precedes_relationships(self, series: str):
        """
        Creates PRECEDES relationships chaining every Document that shares the
        given `series` metadata value (e.g. a recurring meeting name) in
        chronological order by their `date` property.

        Call this once after ALL Documents in the series have been stored —
        e.g. at the end of a batch import — not per-document, since correct
        ordering needs every Document in the series to already be present.
        Requires the caller to have included `series` and `date` in
        `ProcessedDocument.metadata` for each of those documents (see
        `store_chunks_for_doc` / `_create_document_node`); Documents missing
        either field are silently excluded from the chain rather than ordered
        by a guess.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_precedes_relationships,
                series
            )
            logger.info(f"PRECEDES relationships created for series '{series}'")


    def cleanup_singleton_contradictions(self):
        """
        Removes Contradiction nodes with fewer than 2 Description participants.
        See `_cleanup_singleton_contradictions` for why this has to run as a
        post-ingestion pass over the real Neo4j graph rather than inside
        `sanitize_graph`. Call once after all chunks of a document are stored —
        `store_chunks_for_doc` already does this, same as `create_next_relationships`.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._cleanup_singleton_contradictions
            )
            logger.info("Contradiction cleanup pass complete")


    def write_source_metadata(self, source_id: str, props: dict):
        """
        Writes accumulated Source metadata (`date_raw` / `year` / `format`) in one
        explicit post-ingestion SET, bypassing `add_graph_documents` entirely.

        WHY this has to exist as a separate pass: `add_graph_documents` (called
        per-chunk, `baseEntityLabel=False`) compiles to
        `apoc.merge.node([type], {id}, row.properties, {})` — the 4th argument
        (onMatchProperties) is hardcoded to an empty map. So a node's properties
        are only ever set by whichever chunk's write FIRST creates it; every later
        chunk's write for that same id is a MATCH that sets nothing. Source's
        date/year/format are captured progressively across chunks (see
        `sanitize_graph._capture_source_metadata`), so they're lost unless they
        happen to land on the very first chunk that creates the node. This method,
        called once after all per-chunk writes for a document are done, is the
        workaround — see `store_chunks_for_doc` for the call site and where
        `props` is derived.

        Call once per document, after all chunks are stored — same lifetime as
        `cleanup_singleton_contradictions`.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._write_source_metadata,
                source_id,
                props
            )


    def set_chunk_created_at_if_absent(self, chunk_id: str, created_at: str):
        """
        Stamps a Chunk node with the moment it first entered the graph.

        WHY this is a separate guarded pass rather than a `created_at` key in the
        metadata dict handed to `add_embeddings`: that call compiles to
        `MERGE (c:Chunk {id: row.id}) ... SET c += row.metadata` with no
        ON CREATE / ON MATCH split (see Neo4jVector._build_import_query), so a
        metadata entry would be overwritten with "now" on every re-write and the
        timestamp would silently drift to the most recent write instead of
        recording the first. Same class of problem — and same shape of fix — as
        `write_source_metadata`.

        `chunk_id` must be the vector store's node id (the md5 of the chunk
        text), not `Chunk.chunk_id` (the integer position). Take it from
        `add_embeddings`'s return value rather than recomputing the hash, so it
        cannot drift from the library's convention.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._set_chunk_created_at_if_absent,
                chunk_id,
                created_at
            )


    def stamp_entity_created_at(self):
        """
        End-of-document pass: stamps `created_at` on every node in
        `ENTITY_LABELS_NEEDING_CREATED_AT` that doesn't have one yet.

        Call once per document, after all its chunks are stored — same
        lifetime as `write_source_metadata` / `cleanup_singleton_contradictions`
        (see `finalize_document`). One timestamp is reused across every label
        in a single call, so nodes first introduced by the same document's
        ingest share an ingest moment instead of drifting across each label's
        own Cypher round-trip.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        with self._driver.session(database=self._database) as session:
            for label in ENTITY_LABELS_NEEDING_CREATED_AT:
                session.execute_write(
                    self._set_entity_created_at_if_absent,
                    label,
                    created_at
                )
        logger.info(f"Stamped created_at where absent on: {ENTITY_LABELS_NEEDING_CREATED_AT}")


    def create_mentions_relationships(
            self,
            node_id: str,
            chunk_id: int,
            filename: str,
            document_version: int
        ):
        """ Creates MENTIONS relationships between Chunk and __Entity__ nodes. """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_mentions_relationships,
                node_id,
                chunk_id,
                filename,
                document_version
            )
            logger.info(f"MENTIONS relationships created!")


    def store_single_chunk(self, doc: ProcessedDocument, chunk) -> Optional[Tuple[str, dict]]:
        """
        Writes ONE chunk — text, embedding, and whatever graph was extracted from
        it — to Neo4j.

        Split out of `store_chunks_for_doc` so the streaming path
        (`GraphMiner.mine_and_store_doc_chunks`) can write each chunk the moment
        its own extraction is ready, instead of holding a whole document's graph
        in memory until the last chunk is extracted. `store_chunks_for_doc` still
        exists and now just drives this in a loop, so the batch path is unchanged
        for any caller that wants it.

        Returns `(source_node_id, source_properties)` for the Source node this
        chunk carried, or None if it carried none — the caller folds that into
        the end-of-document Source metadata write (see `finalize_document`).
        """
        # Shallow copy, unlike the aliasing `metadata = doc.metadata` this
        # replaced. Two reasons: the streaming caller may invoke this per chunk
        # at arbitrary times, and the old aliasing also meant chunk-level keys
        # (chunk_id, chunk_size, ...) were mutated INTO the document's own
        # metadata dict — which `_create_document_node` then wrote onto the
        # Document node with `SET d += $metadata`. Document nodes will no longer
        # carry the last chunk's chunk_id; that was never intended.
        metadata = {**(doc.metadata or {})}
        metadata["filename"] = doc.filename
        metadata["document_version"] = doc.document_version
        # chunk level metadata
        metadata["chunk_id"] = chunk.chunk_id
        metadata["chunk_size"] = chunk.chunk_size
        metadata["chunk_overlap"] = chunk.chunk_overlap
        metadata["embeddings_model"] = chunk.embeddings_model

        try:
            chunk_node_ids = self.vector_store.add_embeddings(
                texts=[chunk.text],
                embeddings=[chunk.embedding],  # add_embeddings wants one vector per text
                metadatas=[metadata]
            )
            if chunk_node_ids:
                self.set_chunk_created_at_if_absent(
                    chunk_id=chunk_node_ids[0],
                    created_at=datetime.now(timezone.utc).isoformat()
                )
        except Exception as e:
            logger.warning(f"Error storing chunk for document {doc.filename}: {e}")
            return None

        if chunk.nodes is None:
            return None

        source_node: Optional[Tuple[str, dict]] = None
        for node in chunk.nodes:
            if node.type == "Source":
                source_node = (node.id, node.properties)

        graph_doc: GraphDocument = GraphDocument(
            nodes=chunk.nodes,
            relationships=chunk.relationships if chunk.relationships is not None else [],
            source=Document(
                page_content=chunk.text
            )
        )

        try:
            self.add_graph_documents(
                graph_documents=[graph_doc],
                include_source=False,
                baseEntityLabel=False
            )

            for node in chunk.nodes:
                self.create_mentions_relationships(
                    node_id=node.id,
                    chunk_id=chunk.chunk_id,
                    filename=doc.filename,
                    document_version=doc.document_version
                )
        except Exception as e:
            logger.warning(f"Error storing graph for chunk {chunk.chunk_id} in document {doc.filename}: {e}")

        return source_node


    def finalize_document(
        self,
        doc: ProcessedDocument,
        accumulated_source_props: Optional[dict] = None,
        source_node_id: Optional[str] = None
    ):
        """
        Everything that runs ONCE per document, after every one of its chunks is
        in Neo4j: the NEXT chain, the Document node, the Contradiction cleanup,
        the Source metadata write and the vector index.

        Call exactly once, after the last `store_single_chunk` for the document.

        `accumulated_source_props` / `source_node_id` come from the LAST chunk in
        document order that carried a Source node. That relies on an
        implementation detail of `sanitize_graph`: it bakes the accumulated
        `source_meta_state` (date_raw/year/format) into every chunk's own
        Source-node properties, and that accumulation only ever gains keys across
        chunks (never loses them) — so the last one holds the fullest snapshot.
        If `sanitize_graph` ever stops doing that, this silently stops finding
        anything to write and the date quietly stops reaching Neo4j again.
        """
        try:
            self.create_next_relationships(
                filename=doc.filename,
                doc_version=doc.document_version
            )
        except Exception as e:
            logger.warning(f"Error creating NEXT relationships for chunks in Document {doc.filename}: {e}")

        try:
            self.create_document_node(doc=doc)
        except Exception as e:
            logger.warning(f"Error creating Document source node for file: {doc.filename}: {e}")

        try:
            self.cleanup_singleton_contradictions()
        except Exception as e:
            logger.warning(f"Error cleaning up singleton Contradiction nodes for document {doc.filename}: {e}")

        try:
            self.stamp_entity_created_at()
        except Exception as e:
            logger.warning(f"Error stamping created_at on entity nodes for document {doc.filename}: {e}")

        # See `write_source_metadata` docstring for why this explicit pass exists
        # (apoc.merge.node's onMatchProperties is hardcoded empty in add_graph_documents,
        # so only the chunk that first creates the Source node can set its properties).
        if accumulated_source_props and source_node_id:
            source_meta_props = {
                k: v for k, v in accumulated_source_props.items()
                if k in ("date_raw", "year", "format")
            }

            # A date the producer supplied as metadata outranks anything the model
            # read out of the text. For a meeting it is the only date there is —
            # nobody says the date out loud, and the prompt forbids guessing one —
            # and it is exact, being taken from the recording's own id. Without
            # this promotion `Source.year` stays null for every meeting, and the
            # supersession pass skips any pair missing a year on either side
            # (src/conflict/classifier.py), permanently excluding the whole
            # meeting corpus from the one comparison it is best suited for.
            producer_date = (doc.metadata or {}).get("date")
            if producer_date:
                source_meta_props["date_raw"] = str(producer_date)
                # Stored as a string: _Node.properties is Dict[str, str], and
                # fetch_source_years reads it back with toInteger().
                source_meta_props["year"] = str(producer_date)[:4]

            if source_meta_props:
                if "date_raw" in source_meta_props and "year" not in source_meta_props:
                    logger.warning(
                        f"Source '{source_node_id}' has date_raw={source_meta_props['date_raw']!r} "
                        f"but no parseable year — a date was found but couldn't be parsed, "
                        f"as opposed to no date being present at all."
                    )
                try:
                    self.write_source_metadata(source_id=source_node_id, props=source_meta_props)
                except Exception as e:
                    logger.warning(f"Error writing Source metadata for document {doc.filename}: {e}")

        try:
            self.vector_store.create_new_index()
        except Exception as e:
            logger.warning(f"Error creating Index for chunks: {e}")


    def store_chunks_for_doc(self, doc: ProcessedDocument):
        """
        Stores Chunk nodes for a `ProcessedDocument` into the Knowledge Graph and updates the
        Knowledge Graph itself with the graphs extracted from each chunk, if any.

        The batch path: every chunk of the document must already have been
        extracted before this is called. `GraphMiner.mine_and_store_doc_chunks`
        is the streaming alternative, which interleaves extraction and storage so
        a crash costs one chunk instead of the whole document. Both go through
        the same `store_single_chunk` / `finalize_document` pair.
        """
        accumulated_source_props: Optional[dict] = None
        source_node_id: Optional[str] = None

        for chunk in doc.chunks:
            source_node = self.store_single_chunk(doc, chunk)
            if source_node:
                # Later chunks' snapshots are a superset of earlier ones (see
                # `finalize_document`) — overwrite every time.
                source_node_id, accumulated_source_props = source_node

        self.finalize_document(
            doc,
            accumulated_source_props=accumulated_source_props,
            source_node_id=source_node_id
        )


    def add_documents(self, docs: List[ProcessedDocument]):
        for doc in docs:
            self.store_chunks_for_doc(doc)


    def get_digraph(self) -> nx.DiGraph:
        """
        Returns the Knowledge Graph under its `networkx.DiGraph` representation.
        """
        query_nodes = """
            MATCH (n)
            RETURN elementId(n) AS node_id, labels(n) AS labels, properties(n) AS properties;
        """

        query_rels = """
            MATCH (n)-[r]->(m)
            RETURN elementId(n) AS source, elementId(m) AS target, type(r) AS rel_type, properties(r) AS properties;
        """

        G = nx.DiGraph()

        with self._driver.session() as session:

            nodes = session.run(query_nodes)
            for record in nodes:
                G.add_node(record["node_id"], labels=record["labels"], **record["properties"])

            relationships = session.run(query_rels)
            for record in relationships:
                G.add_edge(record["source"], record["target"], type=record["rel_type"], **record["properties"])

        logger.info(f"DiGraph with {len(G.nodes)} nodes and {len(G.edges)} relationships")

        return G


    def update_properties(
        self,
        G: Optional[nx.DiGraph] = None,
        centralities: bool=False,
        leiden_communities: bool=False,
        louvain_communities: bool=False,
        leiden_modularity: Optional[float] = None,
        louvain_modularity: Optional[float] = None,
        ):
        """Update Neo4j nodes with Leiden/Louvain communities and centrality scores"""
        with self._driver.session() as session:

            if any([centralities, leiden_communities, louvain_communities]) == True:

                for node, data in G.nodes(data=True):

                    query, params = build_update_query(
                        node_id=node,
                        centralities=centralities,
                        leiden_communities=leiden_communities,
                        louvain_communities=louvain_communities,
                        community_leiden=int(data.get("community_leiden", -1)),
                        community_louvain=int(data.get("community_louvain", -1)),
                        pagerank=float(data.get("pagerank", 0.0)),
                        betweenness=float(data.get("betweenness", 0.0)),
                        closeness=float(data.get("closeness", 0.0))
                    )
                    try:
                        session.run(query, params)
                    except Exception as e:
                        logger.warning(f"Update Query failed for node_id: {node}")

                logger.info("Updated nodes properties in Graph")

            if leiden_modularity is not None:
                update_modularity(session, leiden_modularity, "leiden")
                logger.info("Updated Leiden Modularity property in Graph")

            if louvain_modularity is not None:
                update_modularity(session, louvain_modularity, "louvain")
                logger.info("Updated Louvain Modularity property in Graph")


    def update_centralities_and_communities(self):
        """
        Computes centralities measures and detects communities in nodes across the Knowledge Graph.
        """

        lv = False
        louvain_mod = None
        ld = False
        leiden_mod = None
        centralities = False

        G = self.get_digraph()

        try:
            G, louvain_mod = detect_louvain_communities(G, return_modularity=True)
            lv = True
        except Exception as e:
            logger.warning(f"Something went wrong detecting Louvain Communities: {e}")

        try:
            G, leiden_mod = detect_leiden_communities(G, return_modularity=True)
            ld = True
        except Exception as e:
            logger.warning(f"Something went wrong detecting Leiden Communities: {e}")

        try:
            G = compute_centralities(G)
            centralities = True
        except Exception as e:
            logger.warning(f"Something went wrong computing Centralities degrees on graph: {e}")

        try:
            self.update_properties(G, centralities, ld, lv, leiden_mod, louvain_mod)
        except Exception as e:
            logger.warning(f"Something went wrong while updating properties on graph nodes: {e}")