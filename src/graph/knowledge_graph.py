import networkx as nx

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from neo4j import ManagedTransaction
from typing import List, Optional, Set

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
        query = """
            CREATE (d:Document {
                filename: $filename,
                document_version: $document_version
            })
            SET d += $metadata
        """
        try:
            tx.run(
                query,
                filename=doc.filename,
                document_version=doc.document_version,
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


    def store_chunks_for_doc(self, doc: ProcessedDocument):
        """
        Stores Chunk nodes for a `ProcessedDocument` into the Knowledge Graph and updates the
        Knowledge Graph itself with the graphs extracted from each chunk, if any.
        """

        # Populated from chunk.nodes below, used for the post-loop Source metadata
        # write. Depends on an implementation detail of `sanitize_graph`: it bakes
        # the accumulated `source_meta_state` (date_raw/year/format) into every
        # chunk's own Source-node properties, and that accumulation only ever
        # gains keys across chunks (never loses them) — so the LAST chunk in doc
        # order that has a Source node carries the fullest snapshot. If
        # `sanitize_graph` ever stops doing that, this silently stops finding
        # anything to write and the date quietly stops reaching Neo4j again.
        accumulated_source_props: Optional[dict] = None
        source_node_id: Optional[str] = None

        for chunk in doc.chunks:

            # doc level metadata
            if doc.metadata:
                metadata = doc.metadata
            else:
                metadata = {}
            metadata["filename"] = doc.filename
            metadata["document_version"] = doc.document_version
            # chunk level metadata
            metadata["chunk_id"] = chunk.chunk_id
            metadata["chunk_size"] = chunk.chunk_size
            metadata["chunk_overlap"] = chunk.chunk_overlap
            metadata["embeddings_model"] = chunk.embeddings_model

            try:
                self.vector_store.add_embeddings(
                    texts=[chunk.text],
                    embeddings=[chunk.embedding],  # add_embeddings wants one vector per text
                    metadatas=[metadata]
                )
            except Exception as e:
                logger.warning(f"Error storing chunk for document {doc.filename}: {e}")

            # store chunk's graph
            if chunk.nodes is not None:

                for node in chunk.nodes:
                    if node.type == "Source":
                        # Later chunks' snapshots are a superset of earlier ones
                        # (see the comment above the loop) — overwrite every time.
                        accumulated_source_props = node.properties
                        source_node_id = node.id

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

        # See `write_source_metadata` docstring for why this explicit pass exists
        # (apoc.merge.node's onMatchProperties is hardcoded empty in add_graph_documents,
        # so only the chunk that first creates the Source node can set its properties).
        if accumulated_source_props and source_node_id:
            source_meta_props = {
                k: v for k, v in accumulated_source_props.items()
                if k in ("date_raw", "year", "format")
            }
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