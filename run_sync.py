"""
Pull chunk files uploaded by a producer (Maruf) from Supabase onto this machine.

    python run_sync.py                     # download everything pending
    python run_sync.py --dry-run           # check what would happen, change nothing
    python run_sync.py --mark-built DOC_ID # after main.py has built that document

Safe to run from cron/launchd every few minutes: it is a single indexed query
when there is nothing to do, and it never starts a KG build.
"""

import argparse
import sys

from dotenv import load_dotenv

from src.sync.supabase_sync import ChunkSync, DOWNLOADED, build_sync_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run(limit: int, dry_run: bool, dest: str, mark_built: str) -> None:
    load_dotenv()
    try:
        conf = build_sync_config(dest_root=dest)
    except RuntimeError as e:
        sys.exit(str(e))  # a missing setting is a note to the operator, not a traceback

    sync = ChunkSync(conf=conf)

    if mark_built:
        moved = sync.mark_built(mark_built)
        logger.info(f"Marked {moved} row(s) for doc_id {mark_built!r} as built.")
        return

    outcomes = sync.run(limit=limit, dry_run=dry_run)
    if not outcomes:
        return

    ok = sum(1 for o in outcomes if o.status == DOWNLOADED)
    logger.info(
        f"Done{' (dry run — nothing written)' if dry_run else ''}. "
        f"{ok} downloaded, {len(outcomes) - ok} failed."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync pending chunk uploads from Supabase into the local chunks_data folder."
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N pending uploads (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Verify and validate, but write nothing and update no rows.")
    parser.add_argument("--dest", default=None, help="Destination root (default: CHUNKS_DEST_DIR or ./chunks_data).")
    parser.add_argument("--mark-built", default="", metavar="DOC_ID", help="Mark a downloaded document as built and exit.")
    args = parser.parse_args()

    run(limit=args.limit, dry_run=args.dry_run, dest=args.dest, mark_built=args.mark_built)


if __name__ == "__main__":
    main()
