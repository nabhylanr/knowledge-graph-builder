-- Supabase schema for the chunk hand-off between a producer (Maruf's
-- academic-pdf-chunker) and knowledge-graph-builder.
--
-- Run once, top to bottom, in the Supabase SQL Editor.
--
-- The design deliberately keeps the *file* in Storage and the *state* in a
-- table. Sync is a poll over this table, not a webhook: state that lives in a
-- row survives the receiving machine being asleep, where a delivered-once event
-- would not. See docs/chunk_sync.md.

-- ---------------------------------------------------------------------------
-- 1. Storage bucket (private — the sync job reads it with the service key)
-- ---------------------------------------------------------------------------

insert into storage.buckets (id, name, public)
values ('chunks', 'chunks', false)
on conflict (id) do nothing;

-- ---------------------------------------------------------------------------
-- 2. Manifest table — one row per uploaded file
-- ---------------------------------------------------------------------------

create table if not exists public.chunk_uploads (
    id            bigserial primary key,

    doc_id        text not null,
    owner         text not null default auth.email(),
    storage_path  text not null,               -- path inside the 'chunks' bucket
    sha256        text not null,               -- of the raw file bytes
    n_chunks      integer,

    -- pending    : uploaded, waiting to be pulled
    -- downloaded : pulled, checksum verified, schema validated, on disk
    -- built      : the KG build has been run for it (set by hand / by the build job)
    -- failed     : checksum or validation failed — see `error`
    status        text not null default 'pending'
                  check (status in ('pending', 'downloaded', 'built', 'failed')),
    error         text,

    uploaded_at   timestamptz not null default now(),
    downloaded_at timestamptz,
    built_at      timestamptz,

    -- Re-uploading identical bytes is a no-op instead of a duplicate build;
    -- a genuinely revised file has a different digest and so is a new row.
    unique (doc_id, sha256)
);

create index if not exists chunk_uploads_status_idx
    on public.chunk_uploads (status, uploaded_at);

-- ---------------------------------------------------------------------------
-- 3. Row-level security
--
-- Producers announce work; they never mark it done. `status` moves only via the
-- service-role key, which bypasses RLS and lives solely on the receiving machine
-- — so a producer cannot claim a file was ingested when it was not.
-- ---------------------------------------------------------------------------

alter table public.chunk_uploads enable row level security;

drop policy if exists "producers insert their own uploads" on public.chunk_uploads;
create policy "producers insert their own uploads"
    on public.chunk_uploads for insert to authenticated
    with check (owner = auth.email());

drop policy if exists "producers read their own uploads" on public.chunk_uploads;
create policy "producers read their own uploads"
    on public.chunk_uploads for select to authenticated
    using (owner = auth.email());

-- No update/delete policy: authenticated users get neither.

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
