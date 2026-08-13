import argparse
import os
from typing import Callable, List, Optional, Tuple

from dotenv import load_dotenv

from src.config import Configuration, EmbedderConf, KnowledgeGraphConfig, LLMConf
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.build_ledger import BuildLedger, content_digest
from src.ingestion.chunk_checkpoint import ChunkCheckpoint
from src.ingestion.chunks_ingestor import ChunksIngestor
from src.ingestion.embedder import ChunkEmbedder
from src.ingestion.graph_miner import GraphMiner
from src.schema import ProcessedDocument
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
            type=os.getenv("RE_MODEL_TYPE", "ollama"),
            model=os.getenv("RE_MODEL_NAME"),
            temperature=os.getenv("RE_MODEL_TEMPERATURE") or 0.0,
            deployment=os.getenv("RE_MODEL_DEPLOYMENT"),
            api_key=os.getenv("RE_API_KEY"),
            endpoint=os.getenv("RE_MODEL_ENDPOINT"),
            api_version=os.getenv("RE_MODEL_API_VERSION") or None,
        ),
    )


def select_unbuilt(
    docs: List[ProcessedDocument],
    knowledge_graph: KnowledgeGraph,
    ledger: BuildLedger,
    rebuild: bool,
) -> List[Tuple[ProcessedDocument, str]]:
    """
    Narrow a freshly loaded batch down to the documents that still need building,
    pairing each with the digest to record once it is in.

    The ledger is reconciled against the graph first: an entry for a document
    Neo4j no longer holds is dropped, so a wiped database rebuilds instead of
    being skipped forever.
    """
    digests = {doc.filename: content_digest(doc) for doc in docs}

    stale = ledger.reconcile(knowledge_graph.document_filenames())
    if stale:
        ledger.save()
        logger.info(f"{len(stale)} ledger entr{'y' if len(stale) == 1 else 'ies'} no longer in the graph — will rebuild: {', '.join(sorted(stale)[:5])}")

    if rebuild:
        logger.warning("--rebuild: ignoring the ledger; documents already in the graph will be added a second time.")
        return [(doc, digests[doc.filename]) for doc in docs]

    pending, skipped = [], []
    for doc in docs:
        if ledger.is_built(doc, digests[doc.filename]):
            skipped.append(doc.filename)
        else:
            pending.append((doc, digests[doc.filename]))

    if skipped:
        logger.info(f"Already built, skipping {len(skipped)} document(s): {', '.join(sorted(skipped)[:5])}"
                    + (f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""))
    return pending


def build_remote_marker(enabled: bool) -> Optional[Callable[[str], None]]:
    """
    A `mark(doc_id)` that moves the producer's manifest row to `built`, or None
    when there is no Supabase configured to tell.

    Best-effort on purpose: the graph is the real outcome and the row is only
    bookkeeping, so a Supabase hiccup must never fail a build that succeeded.
    """
    if not enabled or not (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY")):
        return None

    try:
        from src.sync.supabase_sync import ChunkSync, build_sync_config

        sync = ChunkSync(conf=build_sync_config())
    except Exception as e:
        logger.warning(f"Supabase status updates disabled ({e}).")
        return None

    def mark(doc_id: str) -> None:
        try:
            if sync.mark_built(doc_id):
                logger.info(f"Marked {doc_id} as built in Supabase.")
        except Exception as e:
            logger.warning(f"Could not mark {doc_id} as built in Supabase: {e}")

    return mark


def run(
    chunks_path: str,
    limit: int = 0,
    communities: bool = True,
    rebuild: bool = False,
    ledger_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    remote_status: bool = True,
    corpus_domain: Optional[str] = None,
) -> None:
    conf = build_configuration()

    logger.info(f"Loading chunks from {chunks_path} ...")
    ingestor = ChunksIngestor()
    if os.path.isdir(chunks_path):
        docs = ingestor.load_from_folder(chunks_path)
    else:
        docs = ingestor.load_from_file(chunks_path)

    if not docs:
        logger.warning("No chunks found. Nothing to do.")
        return

    total_chunks = sum(len(d.chunks) for d in docs)
    logger.info(f"Found {total_chunks} chunks across {len(docs)} documents.")

    logger.info("Setting up pipeline components...")
    embedder = ChunkEmbedder(conf=conf.embedder_conf)
    graph_miner = GraphMiner(
        conf=conf.re_model_conf,
        corpus_domain=corpus_domain,
    )
    knowledge_graph = KnowledgeGraph(
        conf=conf.database,
        embeddings_model=embedder.embeddings,
    )

    if not knowledge_graph._driver.verify_authentication():
        logger.error("Could not authenticate against Neo4j — check your NEO4J_* settings.")
        return

    ledger = BuildLedger(path=ledger_path)
    pending = select_unbuilt(docs, knowledge_graph, ledger, rebuild=rebuild)

    # Applied AFTER the ledger filter, so `--limit 2` means "two documents I have
    # not built yet" — repeatable until the folder is exhausted.
    if limit and limit > 0:
        pending = pending[:limit]
        logger.info(f"--limit {limit}: keeping the first {len(pending)} document(s).")

    if not pending:
        logger.info("Every document is already in the graph. Nothing to do.")
        return

    docs = [doc for doc, _ in pending]
    logger.info(f"Building {len(docs)} document(s), {sum(len(d.chunks) for d in docs)} chunks.")

    mark_remote = build_remote_marker(remote_status)
    digests = {doc.filename: digest for doc, digest in pending}
    checkpoint = ChunkCheckpoint(path=checkpoint_path)

    # One document at a time, embedded then extracted-and-stored as it goes,
    # rather than three whole-batch phases. Batching meant nothing reached Neo4j
    # until every pending document had been extracted, so a crash on document 3
    # of 5 also threw away documents 1 and 2. Within a document,
    # `mine_and_store_doc_chunks` writes and checkpoints each chunk as its own
    # extraction lands, so a crash costs the in-flight chunk rather than the
    # document.
    for doc in docs:
        if rebuild:
            # "Rebuild" means redo the work, so stale progress from an earlier
            # run of this document must not be silently consulted.
            checkpoint.clear_document(doc.filename, doc.document_version)

        logger.info(f"Embedding chunks for {doc.filename}...")
        embedder.embed_document_chunks(doc)

        logger.info(f"Extracting and storing {doc.filename}...")
        graph_miner.mine_and_store_doc_chunks(
            doc,
            knowledge_graph=knowledge_graph,
            checkpoint=checkpoint,
        )

        # Recorded per document, not once at the end: an interrupted run should
        # cost the document it died on, not every document it had finished.
        ledger.record(doc, digests.get(doc.filename))
        ledger.save()
        if mark_remote:
            mark_remote(doc.filename)

    # Chain each recurring meeting into a timeline. Runs over the series present
    # in THIS batch but re-chains each one whole, so a meeting added later still
    # links to the ones already in the graph (the Cypher MERGEs over every
    # Document sharing the series, not just the new ones).
    for series in sorted({(doc.metadata or {}).get("series") for doc in docs} - {None, ""}):
        knowledge_graph.create_precedes_relationships(series=series)

    if communities:
        logger.info("Computing centralities and detecting communities...")
        knowledge_graph.update_centralities_and_communities()

    logger.info(
        "Done. Graph now holds "
        f"{knowledge_graph.number_of_nodes} nodes, "
        f"{knowledge_graph.number_of_relationships} relationships, "
        f"{knowledge_graph.number_of_docs} documents."
    )
    knowledge_graph.log_untyped_topic_count()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Neo4j Knowledge Graph from pre-built chunks.")
    parser.add_argument(
        "--chunks",
        default=DEFAULT_CHUNKS_PATH,
        help="Path to a .jsonl file, or a folder searched recursively (default: ./chunks_data).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N not-yet-built documents (0 = all). Handy for a quick test.",
    )
    parser.add_argument(
        "--no-communities",
        action="store_true",
        help="Skip the centralities + community detection step.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Extract every document again, even ones the ledger says are built. "
             "Document nodes are CREATEd, not MERGEd — wipe the graph first, or expect duplicates.",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="Build ledger file (default: BUILD_LEDGER_PATH or ./build_ledger.json).",
    )
    parser.add_argument(
        "--chunk-checkpoint",
        default=None,
        help="Chunk-level progress file, used to resume a document that crashed mid-way "
             "(default: CHUNK_CHECKPOINT_PATH or ./chunk_checkpoint.json).",
    )
    parser.add_argument(
        "--no-remote-status",
        action="store_true",
        help="Do not move the Supabase manifest rows to 'built' after a successful build.",
    )
    parser.add_argument(
        "--corpus-domain",
        choices=["paper", "meeting"],
        default=None,
        help="R19: when this whole build is known to be one domain, restrict the "
             "Type vocabulary shown to the extraction model to that domain only "
             "(e.g. --corpus-domain paper for a paper-only folder like gold_b1). "
             "sanitize_graph's expected_domain guard already GUARANTEES a paper "
             "Topic never links to a meeting Type regardless of this flag — this "
             "just reduces how often that guard has anything to catch. Omit for a "
             "mixed or unclassified folder (unchanged prior behaviour).",
    )
    args = parser.parse_args()

    run(
        chunks_path=args.chunks,
        limit=args.limit,
        communities=not args.no_communities,
        rebuild=args.rebuild,
        ledger_path=args.ledger,
        checkpoint_path=args.chunk_checkpoint,
        remote_status=not args.no_remote_status,
        corpus_domain=args.corpus_domain,
    )


if __name__ == "__main__":
    main()