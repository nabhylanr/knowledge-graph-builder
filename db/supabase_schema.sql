-- Supabase schema for the chunk hand-off between the producers (Maruf's
-- academic-pdf-chunker for papers, the meeting chunker for transcripts) and
-- knowledge-graph-builder.
--
-- For whoever OWNS the Supabase instance. The receiving machine only reads: it
-- needs SUPABASE_URL + SUPABASE_SERVICE_KEY and nothing from this file. On the
-- lab's dev and prod instances these objects already exist.
--
-- Run once, top to bottom, in the Supabase SQL Editor (or over Postgres on
-- :5432 — the service key cannot run DDL, it is a PostgREST token, not a
-- database login).
--
-- Two design choices carry this:
--
--   * The *file* lives in Storage, the *state* in a table. Sync is a poll over
--     that table, not a webhook: state that lives in a row survives the
--     receiving machine being asleep, where a delivered-once event would not.
--
--   * One manifest table per kind of document. The table a row is in IS its
--     doc_type — nothing has to be inferred on the receiving side, the two
--     producers have separate queues (a broken paper upload cannot hold up
--     meetings), and the two queues can carry different file formats: papers
--     arrive as the pipeline's .jsonl, meetings as the chunker's *.chunks.json.
--
-- See docs/chunk_sync.md.

-- ---------------------------------------------------------------------------
-- 1. Storage bucket (private — the sync job reads it with the service key)
--
-- One bucket for both kinds; the manifest table, not the path, says what a file
-- is. Storage is only bytes, so there is nothing to gain from splitting it.
-- ---------------------------------------------------------------------------

insert into storage.buckets (id, name, public)
values ('chunks', 'chunks', false)
on conflict (id) do nothing;

-- ---------------------------------------------------------------------------
-- 2. Manifest tables — one row per uploaded file, one table per doc_type
--
-- Identical shape on purpose: src/sync/supabase_sync.py runs the same query
-- against both and reads the doc_type off the table name.
-- ---------------------------------------------------------------------------

do $$
declare
    manifest text;
begin
    foreach manifest in array array['paper_chunk_uploads', 'meeting_chunk_uploads']
    loop
        execute format($f$
            create table if not exists public.%I (
                id            bigserial primary key,

                doc_id        text not null,
                owner         text not null default auth.email(),
                storage_path  text not null,       -- path inside the 'chunks' bucket
                sha256        text not null,       -- of the raw file bytes
                n_chunks      integer,

                -- pending    : uploaded, waiting to be pulled
                -- downloaded : pulled, checksum verified, schema validated, on disk
                -- built      : the KG build has been run for it (set by main.py / by hand)
                -- failed     : checksum or validation failed — see `error`
                status        text not null default 'pending'
                              check (status in ('pending', 'downloaded', 'built', 'failed')),
                error         text,

                uploaded_at   timestamptz not null default now(),
                downloaded_at timestamptz,
                built_at      timestamptz,

                -- Re-uploading identical bytes is a no-op instead of a duplicate
                -- build; a genuinely revised file has a different digest and so
                -- is a new row.
                unique (doc_id, sha256)
            )
        $f$, manifest);

        execute format(
            'create index if not exists %I on public.%I (status, uploaded_at)',
            manifest || '_status_idx', manifest
        );

        -- ---------------------------------------------------------------
        -- Row-level security
        --
        -- Producers announce work; they never mark it done. `status` moves only
        -- via the service-role key, which bypasses RLS and lives solely on the
        -- receiving machine — so a producer cannot claim a file was ingested
        -- when it was not. No update/delete policy: authenticated users get
        -- neither.
        -- ---------------------------------------------------------------

        execute format('alter table public.%I enable row level security', manifest);

        execute format('drop policy if exists "producers insert their own uploads" on public.%I', manifest);
        execute format($f$
            create policy "producers insert their own uploads"
                on public.%I for insert to authenticated
                with check (owner = auth.email())
        $f$, manifest);

        execute format('drop policy if exists "producers read their own uploads" on public.%I', manifest);
        execute format($f$
            create policy "producers read their own uploads"
                on public.%I for select to authenticated
                using (owner = auth.email())
        $f$, manifest);

        -- ---------------------------------------------------------------
        -- Realtime
        --
        -- Lets run_listen.py have INSERTs pushed to it instead of waiting for
        -- the next poll. This is an optimisation, not the delivery guarantee:
        -- Realtime never replays an event missed while the listener was down,
        -- which is exactly why the scheduled run_sync.py poll stays in place as
        -- the safety net.
        -- ---------------------------------------------------------------

        if not exists (
            select 1 from pg_publication_tables
            where pubname = 'supabase_realtime'
              and schemaname = 'public'
              and tablename = manifest
        ) then
            execute format('alter publication supabase_realtime add table public.%I', manifest);
        end if;
    end loop;
end $$;

-- ---------------------------------------------------------------------------
-- 3. Migration from the single `chunk_uploads` table
--
-- Only fires on a database that still has it: rows are split by their `doc_type`
-- column (rows that never got one follow DEFAULT_DOC_TYPE's old behaviour and
-- go to paper), and the old table is left in place, empty of nothing, for you to
-- drop by hand once you are satisfied.
-- ---------------------------------------------------------------------------

do $$
begin
    if to_regclass('public.chunk_uploads') is null then
        return;
    end if;

    -- The old table may predate its own doc_type column; adding it keeps the
    -- two statements below valid rather than erroring half way through.
    alter table public.chunk_uploads add column if not exists doc_type text;

    insert into public.paper_chunk_uploads
        (doc_id, owner, storage_path, sha256, n_chunks, status, error,
         uploaded_at, downloaded_at, built_at)
    select doc_id, owner, storage_path, sha256, n_chunks, status, error,
           uploaded_at, downloaded_at, built_at
    from public.chunk_uploads
    where coalesce(doc_type, 'paper') = 'paper'
    on conflict (doc_id, sha256) do nothing;

    insert into public.meeting_chunk_uploads
        (doc_id, owner, storage_path, sha256, n_chunks, status, error,
         uploaded_at, downloaded_at, built_at)
    select doc_id, owner, storage_path, sha256, n_chunks, status, error,
           uploaded_at, downloaded_at, built_at
    from public.chunk_uploads
    where doc_type = 'meeting'
    on conflict (doc_id, sha256) do nothing;

    raise notice 'chunk_uploads copied into the per-type tables; drop it when you are happy.';
end $$;

-- ---------------------------------------------------------------------------
-- 4. Storage policies
-- ---------------------------------------------------------------------------

drop policy if exists "producers upload chunk files" on storage.objects;
create policy "producers upload chunk files"
    on storage.objects for insert to authenticated
    with check (bucket_id = 'chunks');

drop policy if exists "producers read their own chunk files" on storage.objects;
create policy "producers read their own chunk files"
    on storage.objects for select to authenticated
    using (bucket_id = 'chunks' and owner = auth.uid());

-- ---------------------------------------------------------------------------
-- 5. After running this
--
--   a. Authentication → Users → "Add user" for each producer (email + password).
--      That account is the trust boundary: only people you create can upload.
--   b. Give the producer SUPABASE_URL + the *anon* key (public by design).
--      Keep the *service_role* key on the receiving machine only — never share it.
-- ---------------------------------------------------------------------------
