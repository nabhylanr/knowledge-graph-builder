# chunks_data/_skip/

Meeting documents held back from the current build, kept with the corpus they
belong to.

## Build from `chunks_data/meeting`, not from `chunks_data`

Nothing in the code skips this folder — the leading underscore is a convention
for readers, not a rule the pipeline enforces. `ChunksIngestor.load_from_folder`
walks its root with `rglob("*.jsonl")` (`src/ingestion/chunks_ingestor.py:103`),
so a build rooted at `chunks_data/` **will** pick these files up.

```
python main.py --chunks chunks_data/meeting     # batch 1 only — use this
python main.py                                  # sweeps _skip back in
```

That first form is already the documented way to build one kind (README:161).
It also means promoting batch 2 needs no file moves at all — just point
`--chunks` at `chunks_data/_skip/meeting-batch2`.

If a hard guarantee is ever wanted instead of a convention, renaming the parked
files so they no longer end in `.jsonl` takes them out of the glob without
touching any code.

## Notes

`src.ingestion.validate` takes a path directly, so parked files can still be
checked while parked.

Safe from re-sync: `src/sync/supabase_sync.py` only downloads rows still marked
`pending` and flips them to `downloaded` once written, so a file already pulled
is never re-fetched into `chunks_data/`.

Restore with `git mv chunks_data/_skip/<sub>/<doc_id>.jsonl chunks_data/meeting/`.
The build ledger keys on `doc_id`, so already-built documents are not rebuilt
when more are added.

| | files | chunks | eligible |
|---|------:|-------:|---------:|
| `chunks_data/meeting/` (building now) | 14 |  656 |  634 |
| `meeting-batch2/`                     | 36 | 1690 | 1383 |
| `meeting-empty/`                      |  6 |   43 |    0 |

---

## meeting-empty/ — six recordings with no speech (parked 2026-08-14)

Every chunk is `[unknown] you`, which is what Whisper emits for silence. None
pass the 70-char eligibility threshold in `src/ingestion/meeting_chunks.py`, so
they already cost no extraction calls — they are parked because they still
created a Document node, embeddings and a `NEXT` chain for content that does not
exist.

| chunks | doc_id |
|-------:|--------|
| 24 | `research-finding-20260708-135551-meeting-recording-ab04d535` |
|  9 | `research-result-report-20260707-054918utc-meeting-recording-0e73dc28` |
|  5 | `rmfs-20260409-170216-meeting-recording-35f6fe9c` |
|  2 | `energy-conference-20260303-084956utc-meeting-recording-a0c02a1e` |
|  2 | `rmfs-meeting-251113-20251113-055232utc-meeting-recording-e71786cc` |
|  1 | `meeting-in-center-for-iot-innovation-citi-general-20260211-140058-meeting-recording-a9774504` |

Almost certainly failed recordings (dead mic, wrong input device), not chunker
bugs. If the audio is recovered and re-transcribed, the new file lands in
`chunks_data/meeting/` as normal and these can be deleted.

Note `rmfs-20260409-170216-meeting-recording-1-960ddd04` — the `-1` companion of
the silent `…35f6fe9c` — has real audio (69/69 eligible) and was **not** parked
here; only the silent half of that session was.

---

## meeting-batch2/ — deferred, not junk (parked 2026-08-14)

The meeting corpus is ~95 h of extraction at the current one-call-per-chunk
rate, so it was split in two rather than built in one run. Batch 1 is what is in
`chunks_data/meeting/` now.

### How batch 1 was chosen

Greedy max-coverage, not "highest density". Each step took the document with the
best *new* idf-weighted vocabulary per eligible chunk, so a meeting that mostly
restates an already-selected one scores low even when its prose is dense. Terms
appearing in only one document (ASR one-offs) or in >80% of them (generic talk)
are excluded from the weighting.

Result: **634 eligible chunks — 31% of the corpus — carrying 79.9% of its total
information weight.**

### Promotion order for batch 2

Returns diminish sharply, which is why batch 1 stops at 14 documents. Restore in
this order to keep every step the best available trade; the percentage is
cumulative coverage of the whole corpus once that document is back in, and the
hours assume ~170 s/chunk (from the one completed build: 5 h / 106 chunks).

| order | eligible | cum. | cum. hours | doc_id |
|------:|---------:|-----:|-----------:|--------|
|  1 | 68 | 82.9% | 33 h | `rmfs-communal-20260226-133203-…9f0fae49` |
|  2 | 14 | 83.6% | 34 h | `ppt-prof-record6-20260122-183937-…f1a02d61` |
|  3 | 44 | 85.0% | 36 h | `utokyo-workshop-20260623-160103-…adab6564` |
|  4 | 14 | 85.5% | 36 h | `meeting-in-general-20260116-161244-…dce9183d` |
|  5 | 62 | 87.4% | 39 h | `llm-bi-weekly-meeting-20260325-140553-…4fdf7e30` |
|  6 | 21 | 88.1% | 40 h | `meeting-in-center-for-iot-innovation-citi-general-20260401-140519-…bd0462aa` |
|  7 | 69 | 90.0% | 43 h | `energy-group-meeting-20260206-142510-920c91cb` |
|  8 | 42 | 90.8% | 45 h | `research-finding-20260708-141120-…e349da70` |
|  9 | 22 | 91.3% | 47 h | `rmfs-meeting-1023-20251023-123751-8c01ff86` |
| 10+ | 1027 | 100% | 95 h | the original 27 deferred documents |

The last nine above were in batch 1 until the corpus was trimmed a second time:
together they cost 356 chunks (~17 h) for 11.4 points of coverage, against the
first fourteen documents' 634 chunks for 79.9.

### What batch 1 is thin on

Coverage was checked per topic afterwards. Batch 1 keeps a real share of every
major thread (lkc 96 mentions vs 161 here, rmfs 40/38, rag 160/153, energy
212/129, robot 244/181, knowledge graph 46/10, reinforcement learning 49/21).
Two land almost entirely here — `retrieval` (19/41) and `embedding` (19/44) —
and `user-testing-presentation-w-prof-hsu` is the corpus's only source of "user
testing". If batch 1's graph looks light on retrieval-side concepts, that is the
split, not a gap in the data.

### Caveat

Document-level information density across this corpus is tight (interquartile
range ~160-190 idf/chunk), so this is **not** a junk/valuable divide — batch 2 is
ordinary content that simply overlaps more with batch 1. Deferring it costs whole
series from the `PRECEDES` timeline until it is merged in: the `lkc-*` update
meetings, four of the six `llm-bi-weekly-meeting` recordings, `world-models`,
`homma`, and `user-testing-presentation-w-prof-hsu` have no representative in
batch 1.
