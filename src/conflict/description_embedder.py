from langchain_core.embeddings import Embeddings
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from src.conflict.config import BlockingConfig
from src.graph.knowledge_graph import KnowledgeGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DescriptionEmbedder:
    """
    Embeds `Description.text` (S2's semantic-blocking input) directly onto the
    existing Description nodes. Mirrors how `ChunkEmbedder` +
    `KnowledgeGraph`'s Chunk `Neo4jVector` work (src/ingestion/embedder.py,
    src/graph/knowledge_graph.py:71-81), pointed at node_label="Description"
    instead of "Chunk", and reusing the caller's `KnowledgeGraph` connection
    (`graph=kg`) instead of re-authenticating with separate credentials.

    Unlike the Chunk path (which lets `add_embeddings` default `ids` to
    md5(text)), this ALWAYS passes the Description's own graph `id` as the
    vector-store id, so `add_embeddings`'s MERGE writes the embedding onto the
    existing Description node rather than creating a separate shadow node
    under a content-hash identity.
    """

    def __init__(self, embeddings: Embeddings, kg: KnowledgeGraph, conf: BlockingConfig):
        self.embeddings = embeddings
        self.kg = kg
        self.conf = conf
        self.vector_store = Neo4jVector(
            embedding=embeddings,
            graph=kg,
            index_name=conf.description_index_name,
            node_label="Description",
            embedding_node_property="embedding",
            text_node_property="text",
        )

    def ensure_index(self) -> None:
        """Create the Description vector index if it doesn't exist yet (idempotent)."""
        dimensions, _ = self.vector_store.retrieve_existing_index()
        if not dimensions:
            self.vector_store.create_new_index()
            logger.info(f"Created vector index '{self.conf.description_index_name}' on :Description(embedding).")
        elif dimensions != self.vector_store.embedding_dimension:
            raise ValueError(
                f"Existing index '{self.conf.description_index_name}' has dimension {dimensions}, "
                f"but the configured embedding model produces {self.vector_store.embedding_dimension}. "
                "Use a different index_name or re-embed with the original model."
            )

    def embed_missing_descriptions(self, force: bool = False) -> int:
        """
        Embeds every allowlisted Description that doesn't have an `embedding`
        yet (or all of them, if `force=True`). Idempotent: MERGE-by-id means
        re-running never duplicates a node, and skipping already-embedded rows
        (the default) means re-running never re-embeds unchanged Descriptions.
        """
        where_embedding = "" if force else "AND d.embedding IS NULL"
        rows = self.kg.query(
            f"""
            MATCH (d:Description)
            WHERE d.typeName IN $allowed_types {where_embedding}
            RETURN d.id AS id, d.text AS text, d.topicName AS topicName
            """,
            params={"allowed_types": self.conf.allowed_types},
        )
        if not rows:
            logger.info("No Descriptions need embedding.")
            return 0

        # What gets EMBEDDED (may be topic-prefixed) is independent from what
        # gets WRITTEN to Description.text — add_embeddings always SETs
        # text_node_property from `texts`, and overwriting the ontology's own
        # `text` property with a topic-prefixed variant would be a knowledge
        # mutation, not embedding maintenance. So `texts=` below is always the
        # untouched original text; only `embed_inputs` varies with the toggle.
        embed_inputs = [
            f"{r['topicName']}: {r['text']}" if self.conf.prepend_topic_name else r["text"]
            for r in rows
        ]
        vectors = self.embeddings.embed_documents(embed_inputs)

        self.vector_store.add_embeddings(
            texts=[r["text"] for r in rows],
            embeddings=vectors,
            ids=[r["id"] for r in rows],
            metadatas=[
                {
                    "embeddings_model": getattr(self.embeddings, "model", None),
                    "embedding_prepend_topic_name": self.conf.prepend_topic_name,
                }
                for _ in rows
            ],
        )
        logger.info(f"Embedded {len(rows)} Description nodes.")
        return len(rows)
