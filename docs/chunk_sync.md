# Chunk Hand-off via Supabase

How chunk files get from a producer's machine (Maruf's `academic-pdf-chunker`)
into this repo's `chunks_data/`.

```
producer                     Supabase                     this machine
────────                     ────────                     ────────────
validate locally  ──┐
upload .jsonl       ├──►  Storage bucket `chunks`
insert manifest row ┘     table  `chunk_uploads`  ◄──  run_sync.py (poll)
                                status=pending          ├ download
                                                        ├ verify sha256
                                                        ├ re-validate schema
                                                        ├ write chunks_data/…
                                status=downloaded  ◄────┘
                                                        main.py  (manual)
                                status=built       ◄──  run_sync.py --mark-built
```

The file lives in Storage; the *state* lives in a row. That is the whole design
decision — see [Why polling](#why-polling-and-not-a-webhook) below.

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

**Receiving side pulls:**

```bash
python run_sync.py              # download everything pending
python run_sync.py --dry-run    # verify + validate, write nothing
```

Then build when you choose to, and close the loop:

```bash
python main.py --chunks chunks_data/maruf
python run_sync.py --mark-built <doc_id>
```

### Running it on a schedule (macOS)

`~/Library/LaunchAgents/com.kg.chunksync.plist`, then
`launchctl load ~/Library/LaunchAgents/com.kg.chunksync.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.kg.chunksync</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/admin/Documents/knowledge-graph-builder/.venv/bin/python</string>
    <string>/Users/admin/Documents/knowledge-graph-builder/run_sync.py</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/admin/Documents/knowledge-graph-builder</string>
  <key>StartInterval</key><integer>600</integer>
  <key>StandardErrorPath</key><string>/tmp/chunksync.err</string>
</dict>
</plist>
```

With nothing pending this is one indexed query — cheap enough to run every 10
minutes forever.

---

## Design notes

### Why polling and not a webhook

A Storage webhook needs Supabase to reach an HTTP endpoint on this machine —
which means an open port, TLS and auth, all of it maintained for a two-person
research project. Realtime over WebSocket avoids the port but needs a daemon
running continuously, and an event that fires while the laptop is asleep is gone
for good, so a reconciliation pass has to exist anyway.

Polling has neither problem, because the state is a row rather than a message.
Asleep for three days? The next run still sees `status='pending'` and picks up
everything. Latency is irrelevant here regardless: a KG build takes hours, so a
10-minute poll interval costs nothing.

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
