# Chunk Hand-off via Supabase

How chunk files get from a producer's machine (Maruf's `academic-pdf-chunker`)
into this repo's `chunks_data/`.

```
producer                     Supabase                     receiving machine
────────                     ────────                     ─────────────────
validate locally  ──┐
upload .jsonl       ├──►  Storage bucket `chunks`
insert manifest row ┘     table  `chunk_uploads`
                                status=pending  ──push──► run_listen.py  (~1s)
                                                ◄──poll── run_sync.py    (10 min)
                                                             ├ download
                                                             ├ verify sha256
                                                             ├ re-validate schema
                                                             ├ write chunks_data/<doc_type>/…
                                status=downloaded  ◄─────────┘
                                                          main.py  (manual)
                                status=built       ◄─────────┘
```

The file lives in Storage; the *state* lives in a row. Both drains call the same
idempotent code and only ever claim rows that are still `pending`.

**The poll is the guarantee; the listener is the optimisation.** See
[Why both](#why-both-a-listener-and-a-poll) below — this is the one thing not to
"simplify" later.

---

## Which instance

This runs against a **self-hosted** Supabase on the lab's Tailscale network, not
supabase.com. Two differences that matter:

| | supabase.com | here (citi-cygnus dev) |
|---|---|---|
| API base URL | `https://<ref>.supabase.co` | `http://100.118.203.111:8080` — Kong serves REST/Auth/Storage on the **same port as Studio**, not 8000 |
| Keys | Project Settings → API | the server's own `.env`, or minted from the `app.settings.jwt_secret` Postgres setting |

The dev instance is shared with another project (bucket
`lab-brain-meeting-audio`); this schema only adds the `chunks` bucket and the
`public.chunk_uploads` table, and touches nothing else.

Prod (`citi-condor`, `100.122.56.39`) is the same shape — same schema, different
host and keys.

## One-time setup (receiving side)

1. **Run the schema.** [`db/supabase_schema.sql`](../db/supabase_schema.sql), via
   the Studio SQL Editor or straight over Postgres on `:5432`. It creates the
   `chunks` bucket, `chunk_uploads`, and the RLS policies. Re-running it is safe.

2. **Create an account per producer.** Studio → Authentication → Users, or the
   admin API with the service key:

   ```bash
   curl -X POST "$SUPABASE_URL/auth/v1/admin/users" \
     -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
     -H "Content-Type: application/json" \
     -d '{"email":"maruf@kg.local","password":"...","email_confirm":true}'
   ```

   The account is the trust boundary: only people you create can upload.

3. **Fill in `.env`** (gitignored — keep it that way):

   ```bash
   SUPABASE_URL=http://100.118.203.111:8080
   SUPABASE_SERVICE_KEY=<service_role JWT>
   SUPABASE_BUCKET=chunks
   CHUNKS_DEST_DIR=chunks_data
   ```

   The **service_role key bypasses RLS entirely**. It stays on this machine —
   never in the repo, never sent to a producer.

4. **Install the client:** `pip install -r requirements.txt`

## One-time setup (producer side)

Give them: the repo (or just `src/ingestion/` + `scripts/upload_chunks.py`), the
API URL, the **anon** key, and their account credentials. They also need to be on
the Tailscale network to reach the host at all.

```bash
pip install supabase pydantic
```

```bash
SUPABASE_URL=http://100.118.203.111:8080
SUPABASE_ANON_KEY=<anon JWT>          # public by design, safe to share
SUPABASE_EMAIL=maruf@kg.local
SUPABASE_PASSWORD=...
```

> Self-hosted does not surface the anon key in a settings page. If it is not in
> the server's `.env` as `ANON_KEY`, mint one — it is just an HS256 JWT with
> `{"role":"anon","iss":"supabase","iat":…,"exp":…}` signed with the same secret
> as the service key (`select current_setting('app.settings.jwt_secret')`).
> Kong accepts any correctly signed key; the `role` claim is what limits it.

---

## Daily use

**Producer uploads:**

```bash
python scripts/upload_chunks.py out/*.jsonl
python scripts/upload_chunks.py out/ --dry-run    # validate only
```

Every file is validated against [chunk_schema.md](./chunk_schema.md) first, and
**a batch with any invalid file uploads nothing**. Files land at
`chunks/<producer>/<doc_id>__<sha8>.jsonl`.

### Where a file lands locally

The receiving side files documents by **what they are**, not by who sent them:

```
chunks_data/
├── paper/      # theses, articles — anything with a bibliography
└── meeting/    # transcripts, minutes
```

`resolve_doc_type` ([supabase_sync.py](../src/sync/supabase_sync.py)) picks the
folder from the first of these that answers, and logs which one did:

| # | Signal | Set by |
|---|--------|--------|
| 1 | `doc_type` column on the manifest row (`paper` \| `meeting`) | the producer, explicitly — **preferred** |
| 2 | a `paper/` or `meeting/` prefix on the bucket path | the producer's upload path |
| 3 | the file's own `source_kind`/`source_type` (`pdf` → paper, `transcript` → meeting) | already in the chunk records |
| 4 | `DEFAULT_DOC_TYPE` (`paper`) | nothing — logged as a fallback |

Signals 2–4 exist so nothing breaks before a producer starts sending `doc_type`;
a guess is still a guess, so **set the column**. The receiver writes the
resolved value back onto the row (when the column exists), which makes a
misfiled document visible in the manifest rather than only on disk.

The producer's own folder name is not mirrored — only the file's base name is
kept.

**Receiving side pulls:**

```bash
python run_sync.py              # download everything pending, then exit
python run_sync.py --dry-run    # verify + validate, write nothing
python run_listen.py            # stay connected; pull each upload as it lands
```

`run_listen.py` sweeps the queue on startup and on every (re)subscribe, so
starting it after a period offline picks up whatever accumulated.

Then build when you choose to:

```bash
python main.py                      # every folder under chunks_data, minus what is already built
python main.py --chunks chunks_data/meeting
```

`main.py` records each document in the build ledger as it lands in Neo4j and
moves its manifest row to `built` — so re-running it extracts only what is new.
See [Not building the same thing twice](#not-building-the-same-thing-twice).
`python run_sync.py --mark-built <doc_id>` still exists for closing the loop by
hand (e.g. after a build run with `--no-remote-status`).

### Not building the same thing twice

`main.py` keeps a **build ledger** — `build_ledger.json` in the repo root
(gitignored; override with `--ledger` or `BUILD_LEDGER_PATH`). Every document
that reaches Neo4j is recorded there, and a later run loads the whole folder but
extracts only what is missing.

```bash
python -m src.ingestion.build_ledger              # what is already in the graph
python -m src.ingestion.build_ledger --forget ID  # force one document to rebuild
python main.py --rebuild                          # ignore the ledger entirely
```

This matters more than "don't waste an afternoon": `_create_document_node` uses
`CREATE`, not `MERGE`, so a second run over the same folder produces a **second
Document node** for the same file. The ledger is what makes `python main.py`
safe to just run again.

Three details worth knowing:

- **Documents are keyed by content, not file name.** The digest covers the chunk
  ids and their text, so the same document re-uploaded under a new name is not
  rebuilt, while genuinely revised text is.
- **The ledger is reconciled against Neo4j on every run.** An entry whose
  `Document` node is gone is dropped automatically — so `MATCH (n) DETACH DELETE
  n` refills as expected instead of skipping everything.
- **It is written per document**, right after that document lands in Neo4j.
  An interrupted run costs the document it died on, not the ones that finished.

`--limit` applies *after* the ledger filter: `python main.py --limit 2` means
"build two documents I have not built yet", so it can be repeated until the
folder is exhausted.

### Running it on a schedule

With nothing pending the poll is a single indexed query — cheap enough to run
every 10 minutes forever, on either platform.

**Windows (the deployment target).** Install both — the listener for speed, the
poll as the net:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\install_chunksync_task.ps1      # kg-chunksync
powershell -ExecutionPolicy Bypass -File deploy\install_chunksync_listener.ps1  # kg-chunksync-listener
```

Add `-AsSystem` to either so they keep running with nobody logged on. Logs land
in `chunksync.log` and `chunksync-listener.log` in the repo root.

The poll task runs `deploy\chunksync.cmd` every 10 minutes; two settings carry
the design: `-StartWhenAvailable` runs a poll missed while the machine was off,
and `IgnoreNew` stops a slow run stacking on itself. With the listener installed
the poll is pure backstop, so `-IntervalMinutes 30` is plenty.

The listener task runs at startup (or logon) with no execution time limit and
restarts on failure — Task Scheduler is used instead of NSSM/winsw so there is
nothing extra to install.

```powershell
Start-ScheduledTask     -TaskName kg-chunksync    # run once now
Get-ScheduledTaskInfo   -TaskName kg-chunksync    # last run time + result
Unregister-ScheduledTask -TaskName kg-chunksync -Confirm:$false
```

**macOS (dev machines):** a launchd agent at
`~/Library/LaunchAgents/com.kg.chunksync.plist`, logging to
`~/Library/Logs/chunksync.log`.

```bash
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.kg.chunksync.plist
launchctl kickstart gui/$UID/com.kg.chunksync    # run once now
launchctl bootout   gui/$UID/com.kg.chunksync    # remove
```

> **Only one machine may poll a given Supabase instance.** The first syncer to
> see a row marks it `downloaded`, and the second then finds nothing pending —
> so the file lands on whichever machine happened to poll first. If a Mac is
> developing against dev while Windows deploys against dev too, disable one of
> them, or point them at separate instances (dev vs prod).

---

## Design notes

### Why both a listener and a poll

A Storage webhook was ruled out first: it needs Supabase to reach an HTTP
endpoint on the receiving machine — an open port, TLS and auth to maintain.
Realtime over WebSocket avoids all of that, and is what `run_listen.py` uses.

But **Realtime cannot be the delivery guarantee**, because it never replays an
event that fired while the listener was down. That is not a theoretical worry —
it was verified: kill the daemon, upload a file, and no event is ever delivered
for it. What recovers the file is the sweep, not the socket.

So the two mechanisms have different jobs:

| | `run_listen.py` | `run_sync.py` |
|---|---|---|
| Job | speed (~1s pickup) | the actual guarantee |
| Shape | long-lived daemon | runs and exits |
| If it dies | latency degrades to the poll interval | uploads sit unnoticed |

They are separate processes on purpose. Folding the poll *into* the daemon would
look tidier, but then a dead daemon means nothing polls at all — which is exactly
the failure the poll exists to cover.

Both can drain the same row simultaneously; that is harmless. The download is
idempotent, the bytes are identical, and the status update is a no-op the second
time. The one real hazard — two writers sharing a temp filename — is why
`_write_atomically` puts the pid in the temp name.

### Why the event is never trusted as data

The listener ignores the payload of the INSERT it receives and re-queries for
everything `pending`. The event is a *wake-up*, nothing more. A duplicated,
reordered or malformed event therefore cannot corrupt anything — at worst it
triggers a query that finds no work.

### Why sync never triggers a build

`run_sync.py` stops at `downloaded`. If cron started `main.py`, two uploads
arriving minutes apart would launch two multi-hour extraction runs competing for
the same Ollama backend and Neo4j instance — and you would not find out until
long after. Downloading is safe to automate; building is a decision.

### Why the producer cannot set `status`

RLS grants `authenticated` only INSERT and SELECT on their own rows. `status`
moves only under the service-role key, which exists on the receiving machine
alone — so "it was ingested" is always something this side observed, never
something the other side asserted.

### Why the file is validated twice

Once by the producer (fast feedback, on the machine that can fix it) and once
after download (the uploader could be an older version, or the bytes could have
been mangled). The second pass writes its failure into the row's `error` column,
which the producer can read on their own dashboard — no back-and-forth needed.

### Checksums

`sha256` is computed before upload and re-checked after download; a mismatch
fails the row instead of writing the file. `unique (doc_id, sha256)` makes
re-uploading identical bytes a no-op, while a genuinely revised file gets a new
digest and therefore a new row.
