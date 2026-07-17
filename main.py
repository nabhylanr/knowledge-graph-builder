"""
Build a Knowledge Graph in Neo4j out of a pre-built chunks dataset.

Pipeline (no UI, no document loading — chunks are read straight from disk):

    load chunks (.jsonl)  ->  embed  ->  extract graph (LLM)  ->  store in Neo4j
                                                                     -> centralities & communities

Usage:
    python main.py                       # ingest everything under ./chunks_data
    python main.py --chunks path/to.jsonl
    python main.py --limit 2             # only the first 2 documents (quick test)
    python main.py --no-communities      # skip centralities/communities step
"""

import argparse
import os

from dotenv import load_dotenv

from src.config import Configuration, EmbedderConf, KnowledgeGraphConfig, LLMConf
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.chunks_ingestor import ChunksIngestor
from src.ingestion.embedder import ChunkEmbedder
from src.ingestion.graph_miner import GraphMiner
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHUNKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunks_data")


def build_configuration() -> Configuration:
    """Builds the pipeline `Configuration` from environment variables (.env)."""
    load_dotenv()

    return Configuration(
        database=KnowledgeGraphConfig(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name=os.getenv("INDEX_NAME") or "vector",
        ),
        embedder_conf=EmbedderConf(
            type=os.getenv("EMBEDDINGS_TYPE", "ollama"),
            model=os.getenv("EMBEDDINGS_MODEL_NAME"),
            api_key=os.getenv("EMBEDDINGS_API_KEY"),
            deployment=os.getenv("EMBEDDINGS_DEPLOYMENT"),
            endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
            api_version=os.getenv("EMBEDDINGS_API_VERSION"),
        ),
        re_model_conf=LLMConf(
            type=os.getenv("RE_MODEL_TYPE", "groq"),
            model=os.getenv("RE_MODEL_NAME"),
            temperature=os.getenv("RE_MODEL_TEMPERATURE") or 0.0,
            deployment=os.getenv("RE_MODEL_DEPLOYMENT"),
            api_key=os.getenv("RE_API_KEY"),
            endpoint=os.getenv("RE_MODEL_ENDPOINT"),
            api_version=os.getenv("RE_MODEL_API_VERSION") or None,
        ),
    )


def run(chunks_path: str, limit: int = 0, communities: bool = True) -> None:
    conf = build_configuration()

    logger.info(f"Loading chunks from {chunks_path} ...")
    ingestor = ChunksIngestor()
    if os.path.isdir(chunks_path):
        docs = ingestor.load_from_folder(chunks_path)
    else:
        docs = ingestor.load_from_file(chunks_path)

    if limit and limit > 0:
        docs = docs[:limit]
        logger.info(f"--limit {limit}: keeping the first {len(docs)} document(s).")

    if not docs:
        logger.warning("No chunks found. Nothing to do.")
        return

    total_chunks = sum(len(d.chunks) for d in docs)
    logger.info(f"Found {total_chunks} chunks across {len(docs)} documents.")

    logger.info("Setting up pipeline components...")
    embedder = ChunkEmbedder(conf=conf.embedder_conf)
    graph_miner = GraphMiner(
        conf=conf.re_model_conf,
        ontology=conf.database.ontology,
    )
    knowledge_graph = KnowledgeGraph(
        conf=conf.database,
        embeddings_model=embedder.embeddings,
    )

    if not knowledge_graph._driver.verify_authentication():
        logger.error("Could not authenticate against Neo4j — check your NEO4J_* settings.")
        return

    logger.info("Embedding chunks...")
    docs = embedder.embed_documents_chunks(docs)

    logger.info("Extracting a Knowledge Graph from each chunk...")
    docs = graph_miner.mine_graph_from_docs(docs=docs)

    logger.info("Uploading nodes, relationships and chunks to Neo4j...")
    knowledge_graph.add_documents(docs)

    if communities:
        logger.info("Computing centralities and detecting communities...")
        knowledge_graph.update_centralities_and_communities()

    logger.info(
        "Done. Graph now holds "
        f"{knowledge_graph.number_of_nodes} nodes, "
        f"{knowledge_graph.number_of_relationships} relationships, "
        f"{knowledge_graph.number_of_docs} documents."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Neo4j Knowledge Graph from pre-built chunks.")
    parser.add_argument(
        "--chunks",
        default=DEFAULT_CHUNKS_PATH,
        help="Path to a .jsonl file or a folder of .jsonl files (default: ./chunks_data).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N documents (0 = all). Handy for a quick test.",
    )
    parser.add_argument(
        "--no-communities",
        action="store_true",
        help="Skip the centralities + community detection step.",
    )
    args = parser.parse_args()

    run(chunks_path=args.chunks, limit=args.limit, communities=not args.no_communities)


if __name__ == "__main__":
    main()
