"""
Listen for chunk uploads and pull them the moment they land.

    python run_listen.py

Long-lived: run it as a service (see deploy/install_chunksync_listener.ps1 on
Windows). It is the *fast* half of the hand-off — the scheduled `run_sync.py`
poll stays in place as the safety net, because Realtime never replays events
missed while this process was down.

Like run_sync.py it only downloads; it never starts a KG build.
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from src.sync.realtime_listener import ChunkListener
from src.sync.supabase_sync import build_sync_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listen for Supabase chunk uploads and sync them immediately."
    )
    parser.add_argument("--dest", default=None, help="Destination root (default: CHUNKS_DEST_DIR or ./chunks_data).")
    parser.add_argument(
        "--sweep-interval",
        type=int,
        default=900,
        help="Seconds between in-process reconciliation sweeps (default 900). 0 disables them.",
    )
    args = parser.parse_args()

    load_dotenv()
    try:
        conf = build_sync_config(dest_root=args.dest)
    except RuntimeError as e:
        sys.exit(str(e))

    listener = ChunkListener(conf=conf, sweep_interval=args.sweep_interval or 10**9)
    try:
        asyncio.run(listener.run_forever())
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
