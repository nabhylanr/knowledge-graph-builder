"""
Realtime listener — the fast half of the hybrid hand-off.

Subscribes to INSERTs on every manifest table (`paper_chunk_uploads`,
`meeting_chunk_uploads`) over Supabase Realtime and drains the pending queue
within a second or so of a producer uploading, instead of waiting for the next
scheduled poll.

Two properties are deliberate:

* **The event is only a wake-up, never the data.** The handler ignores the
  payload and re-queries for everything `pending`. So a malformed, duplicated or
  out-of-order event cannot corrupt anything — worst case it triggers a query
  that finds nothing.
* **This does not replace `run_sync.py`.** Realtime never replays an event that
  fired while the listener was down, so the scheduled poll stays as the safety
  net. Losing this daemon costs latency, never data.

Runs on the async client on purpose: in realtime 2.31 the *sync* channel is a
stub with no `on_postgres_changes`, and `AsyncRealtimeClient.listen()` is a
no-op — the socket is pumped by a task that `connect()` starts, so this module
supervises the session itself rather than awaiting `listen()`.
"""

import asyncio
from typing import Any, Dict, Optional

from src.sync.supabase_sync import ChunkSync, SyncConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHANNEL_TOPIC = "chunk-uploads"
REALTIME_PATH = "/realtime/v1"

# How often to notice the socket has dropped. The client heartbeats every 25s,
# so anything well under that reacts promptly without busy-waiting.
CONNECTION_POLL_SECONDS = 5

# Reconnect backoff bounds for the outer supervisor.
INITIAL_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 60


class ChunkListener:
    """
    Long-lived daemon: listen for INSERTs, drain the queue when one arrives.

    `sweep_interval` is an in-process periodic drain that covers the case where
    the socket looks healthy but events silently stop arriving. It is a second
    line of defence, not the safety net — that is the external scheduled poll.
    """

    def __init__(self, conf: SyncConfig, sweep_interval: int = 900):
        self.conf = conf
        self.sync = ChunkSync(conf)
        self.sweep_interval = sweep_interval
        self._wake = asyncio.Event()

    @property
    def endpoint(self) -> str:
        """http://host:8080 -> ws://host:8080/realtime/v1 (https -> wss)."""
        return self.conf.url.rstrip("/").replace("http", "ws", 1) + REALTIME_PATH

    # -- callbacks (run on the event loop; must not block) -------------------

    def _on_insert(self, payload: Dict[str, Any]) -> None:
        doc_id = "?"
        try:  # purely for the log line — the sweep does not depend on it
            doc_id = payload["data"]["record"].get("doc_id", "?")
        except Exception:
            pass
        logger.info(f"Realtime: new upload announced ({doc_id}) — draining queue")
        self._wake.set()

    def _on_state(self, state: Any, error: Optional[Exception]) -> None:
        logger.info(f"Realtime subscription state: {state}{f' ({error})' if error else ''}")
        if str(getattr(state, "value", state)).upper() == "SUBSCRIBED":
            # Anything uploaded while the socket was down produced no event we
            # will ever see, so treat every successful (re)subscribe as a reason
            # to reconcile.
            self._wake.set()

    # -- workers ------------------------------------------------------------

    async def _drain_worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                # ChunkSync is blocking (httpx sync + file IO); keep it off the
                # event loop so heartbeats keep flowing during a long download.
                await loop.run_in_executor(None, self.sync.run)
            except Exception as e:
                logger.warning(f"Sync run failed, will retry on the next trigger: {e}")

    async def _sweep_ticker(self) -> None:
        while True:
            await asyncio.sleep(self.sweep_interval)
            logger.info("Periodic sweep.")
            self._wake.set()

    # -- session / supervisor ------------------------------------------------

    async def _session(self) -> None:
        """One connected session. Returns when the socket drops."""
        from realtime import AsyncRealtimeClient

        client = AsyncRealtimeClient(self.endpoint, self.conf.service_key, auto_reconnect=True)
        await client.connect()

        # One channel, one binding per table. They share a handler because the
        # event is only a wake-up: the drain re-queries every table anyway, so
        # which one fired does not matter.
        channel = client.channel(CHANNEL_TOPIC)
        tables = sorted(set(self.conf.tables.values()))
        for table in tables:
            channel.on_postgres_changes(
                "INSERT",
                callback=self._on_insert,
                schema="public",
                table=table,
            )
        await channel.subscribe(self._on_state)
        logger.info(f"Listening on {self.endpoint} for INSERTs on {', '.join(tables)}.")

        try:
            while client.is_connected:
                await asyncio.sleep(CONNECTION_POLL_SECONDS)
            logger.warning("Realtime socket dropped.")
        finally:
            try:
                await client.close()
            except Exception:
                pass

    async def run_forever(self) -> None:
        workers = [
            asyncio.create_task(self._drain_worker()),
            asyncio.create_task(self._sweep_ticker()),
        ]
        # Drain once before any event arrives — the queue may already hold work
        # from while this daemon was not running.
        self._wake.set()

        backoff = INITIAL_BACKOFF_SECONDS
        try:
            while True:
                try:
                    await self._session()
                    backoff = INITIAL_BACKOFF_SECONDS
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"Listener session ended: {e}")
                logger.info(f"Reconnecting in {backoff}s ...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
