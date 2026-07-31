import argparse
import os

from dotenv import load_dotenv

from src.config import EmbedderConf, KnowledgeGraphConfig
from src.conflict.candidate_store import CandidateStore
from src.conflict.config import DEFAULT_GENERIC_TOPIC_BLOCKLIST, GateConfig
from src.conflict.gates import run_gates
from src.factory.embeddings import get_embeddings
from src.graph.knowledge_graph import KnowledgeGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_gate_config() -> GateConfig:
    load_dotenv()
    blocklist_env = os.getenv("GATE_GENERIC_TOPIC_BLOCKLIST")
    return GateConfig(
        min_text_length=int(os.getenv("GATE_MIN_TEXT_LENGTH", "40")),
        generic_topic_blocklist=[t.strip() for t in blocklist_env.split(",")] if blocklist_env else DEFAULT_GENERIC_TOPIC_BLOCKLIST,
        intra_doc_similarity_threshold=float(os.getenv("GATE_INTRA_DOC_SIMILARITY_THRESHOLD", "0.9")),
        nli_model=os.getenv("GATE_NLI_MODEL", "qwen2.5:3b"),
        nli_reject_confidence=float(os.getenv("GATE_NLI_REJECT_CONFIDENCE", "0.8")),
        nli_concurrency=int(os.getenv("GATE_NLI_CONCURRENCY", "2")),
        nli_timeout_seconds=int(os.getenv("GATE_NLI_TIMEOUT_SECONDS", "30")),
        gate_version=os.getenv("GATE_VERSION", "gate-v1"),
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
    # clean instead of relying on the swallowed-exception None-embeddings path.
    return EmbedderConf(
        type=os.getenv("EMBEDDINGS_TYPE", "ollama"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME"),
        api_key=os.getenv("EMBEDDINGS_API_KEY"),
        deployment=os.getenv("EMBEDDINGS_DEPLOYMENT"),
        endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
        api_version=os.getenv("EMBEDDINGS_API_VERSION"),
    )


def run(sqlite_path: str) -> None:
    conf = build_gate_config()
    embeddings = get_embeddings(build_embedder_config())
    kg = KnowledgeGraph(conf=build_db_config(), embeddings_model=embeddings)

    if not kg._driver.verify_authentication():
        logger.error("Could not authenticate against Neo4j — check your NEO4J_* settings.")
        return

    with CandidateStore(sqlite_path) as store:
        stats = run_gates(kg=kg, store=store, conf=conf)
        logger.info(f"By gate status: {store.count_by_gate_status()}")
        logger.info(f"By rejection reason: {store.count_by_gate_reason()}")

    logger.info(f"Done. {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 (gate filtering) of the whole-KB conflict-detection pass."
    )
    parser.add_argument(
        "--sqlite-path",
        default=os.getenv("BLOCKING_SQLITE_PATH", "conflict_candidates.db"),
        help="Same SQLite file run_blocking.py wrote candidates into.",
    )
    args = parser.parse_args()
    run(sqlite_path=args.sqlite_path)


if __name__ == "__main__":
    main()
