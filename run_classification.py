import argparse
import os

from dotenv import load_dotenv

from src.config import EmbedderConf, KnowledgeGraphConfig
from src.conflict.candidate_store import CandidateStore
from src.conflict.classifier import run_classification
from src.conflict.config import ClassificationConfig, DEFAULT_ALLOWED_TYPES, resolve_ollama_endpoint
from src.factory.embeddings import get_embeddings
from src.graph.knowledge_graph import KnowledgeGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_classification_config() -> ClassificationConfig:
    load_dotenv()
    allowed_types_env = os.getenv("CLASSIFICATION_ALLOWED_TYPES")
    model_type = os.getenv("CLASSIFICATION_MODEL_TYPE", "ollama")
    model = os.getenv("CLASSIFICATION_MODEL_NAME", "qwen3:4b")
    endpoint = resolve_ollama_endpoint(os.getenv("CLASSIFICATION_MODEL_ENDPOINT"))
    logger.info(f"Classification model resolved to: model_type={model_type!r} model={model!r} endpoint={endpoint!r}")
    return ClassificationConfig(
        model_type=model_type,
        model=model,
        temperature=float(os.getenv("CLASSIFICATION_TEMPERATURE", "0.0")),
        api_key=os.getenv("CLASSIFICATION_API_KEY") or os.getenv("RE_API_KEY"),
        endpoint=endpoint,
        max_retries=int(os.getenv("CLASSIFICATION_MAX_RETRIES", "5")),
        retry_wait_seconds=int(os.getenv("CLASSIFICATION_RETRY_WAIT_SECONDS", "65")),
        conn_retry_wait_seconds=int(os.getenv("CLASSIFICATION_CONN_RETRY_WAIT_SECONDS", "30")),
        allowed_types=[t.strip() for t in allowed_types_env.split(",")] if allowed_types_env else DEFAULT_ALLOWED_TYPES,
        partial_context_confidence_factor=float(os.getenv("CLASSIFICATION_PARTIAL_CONTEXT_FACTOR", "0.7")),
        max_cluster_size=int(os.getenv("CLASSIFICATION_MAX_CLUSTER_SIZE", "5")),
        pipeline_version=os.getenv("CLASSIFICATION_PIPELINE_VERSION", "classification-v2"),
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
    # KnowledgeGraph's __init__ unconditionally wires up a Chunk Neo4jVector,
    # which needs a real Embeddings instance — this stage never queries it, but
    # constructing one (cheap, no API call) keeps KnowledgeGraph construction
    # clean instead of relying on the swallowed-exception None-embeddings path
    # (same reasoning as run_gates.py).
    return EmbedderConf(
        type=os.getenv("EMBEDDINGS_TYPE", "ollama"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME"),
        api_key=os.getenv("EMBEDDINGS_API_KEY"),
        deployment=os.getenv("EMBEDDINGS_DEPLOYMENT"),
        endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
        api_version=os.getenv("EMBEDDINGS_API_VERSION"),
    )


def run(sqlite_path: str) -> None:
    conf = build_classification_config()
    embeddings = get_embeddings(build_embedder_config())
    kg = KnowledgeGraph(conf=build_db_config(), embeddings_model=embeddings)

    if not kg._driver.verify_authentication():
        logger.error("Could not authenticate against Neo4j — check your NEO4J_* settings.")
        return

    with CandidateStore(sqlite_path) as store:
        stats = run_classification(kg=kg, store=store, conf=conf)
        logger.info(f"Clusters by resolution_type: {store.count_clusters_by_resolution_type()}")
        logger.info(f"Clusters by outcome: {store.count_clusters_by_outcome()}")
        logger.info(f"Pairs by supersession result: {store.count_by_supersession_result()}")
        logger.info(f"Supersedes by basis: {store.count_supersession_by_basis()}")

    logger.info(f"Done. {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 (classification & write-back) of the whole-KB conflict-detection pass."
    )
    parser.add_argument(
        "--sqlite-path",
        default=os.getenv("BLOCKING_SQLITE_PATH", "conflict_candidates.db"),
        help="Same SQLite file run_blocking.py / run_gates.py wrote candidates into.",
    )
    args = parser.parse_args()
    run(sqlite_path=args.sqlite_path)


if __name__ == "__main__":
    main()
