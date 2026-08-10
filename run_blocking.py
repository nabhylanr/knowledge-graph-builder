import argparse
import os

from dotenv import load_dotenv

from src.config import EmbedderConf, KnowledgeGraphConfig
from src.conflict.blocking import generate_candidates
from src.conflict.candidate_store import CandidateStore
from src.conflict.config import BlockingConfig, DEFAULT_ALLOWED_TYPES
from src.conflict.description_embedder import DescriptionEmbedder
from src.factory.embeddings import get_embeddings
from src.graph.knowledge_graph import KnowledgeGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_blocking_config() -> BlockingConfig:
    load_dotenv()
    allowed_types_env = os.getenv("BLOCKING_ALLOWED_TYPES")
    return BlockingConfig(
        k=int(os.getenv("BLOCKING_K", "15")),
        min_similarity=float(os.getenv("BLOCKING_MIN_SIMILARITY", "0.85")),
        allowed_types=[t.strip() for t in allowed_types_env.split(",")] if allowed_types_env else DEFAULT_ALLOWED_TYPES,
        prepend_topic_name=os.getenv("BLOCKING_PREPEND_TOPIC_NAME", "false").lower() == "true",
        description_index_name=os.getenv("BLOCKING_DESCRIPTION_INDEX_NAME", "description_vector"),
        pipeline_version=os.getenv("BLOCKING_PIPELINE_VERSION", "blocking-v1"),
        sqlite_path=os.getenv("BLOCKING_SQLITE_PATH", "conflict_candidates.db"),
    )


def build_db_config() -> KnowledgeGraphConfig:
    return KnowledgeGraphConfig(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        index_name=os.getenv("INDEX_NAME") or "vector",
    )


def build_embedder_config() -> EmbedderConf:
    return EmbedderConf(
        type=os.getenv("EMBEDDINGS_TYPE", "ollama"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME"),
        api_key=os.getenv("EMBEDDINGS_API_KEY"),
        deployment=os.getenv("EMBEDDINGS_DEPLOYMENT"),
        endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
        api_version=os.getenv("EMBEDDINGS_API_VERSION"),
    )


def run(force_reembed: bool = False) -> None:
    conf = build_blocking_config()
    embeddings = get_embeddings(build_embedder_config())
    kg = KnowledgeGraph(conf=build_db_config(), embeddings_model=embeddings)

    if not kg._driver.verify_authentication():
        logger.error("Could not authenticate against Neo4j — check your NEO4J_* settings.")
        return

    embedder = DescriptionEmbedder(embeddings=embeddings, kg=kg, conf=conf)
    embedder.ensure_index()
    n_embedded = embedder.embed_missing_descriptions(force=force_reembed)
    logger.info(f"Embedded {n_embedded} Description node(s).")

    with CandidateStore(conf.sqlite_path) as store:
        stats = generate_candidates(kg=kg, conf=conf, store=store)
        logger.info(f"Total candidate rows in {conf.sqlite_path}: {store.count()}")
        logger.info(f"By strategy: {store.count_by_strategy()}")

    logger.info(f"Done. {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 (candidate generation / blocking) of the whole-KB conflict-detection pass."
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Re-embed every allowlisted Description, even ones that already have an embedding.",
    )
    args = parser.parse_args()
    run(force_reembed=args.force_reembed)


if __name__ == "__main__":
    main()
