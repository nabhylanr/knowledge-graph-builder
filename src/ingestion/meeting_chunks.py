"""
Adapter for the meeting chunker's `*.chunks.json` dialect.

That producer emits ONE JSON object holding a `chunks` array
(`input_schema_version: speaker_transcript.v1`), which the ingestor cannot read:
it expects one record per line, and it never sees `meeting_id`.

This is deliberately an *adapter* sitting beside the ingestor rather than part of
it — it translates one producer's dialect, while `ChunkRecord` is the contract
every producer must land on. It lives here rather than in `scripts/` only because
two callers need it: `scripts/convert_meeting_chunks.py` for files already on
disk, and `src.sync.supabase_sync` for files pulled out of Supabase.

Three mappings do the real work:

- `meeting_id` -> `doc_id`. Without it every chunk falls back to
  `Path(source_path).stem`, which for these files is the shared
  `speaker_transcript` — i.e. every meeting collapsing into one Document.
- array position -> `index`. The producer's own `chunk_id` is the string
  `"<meeting>:chunk:0007"`, and a string id builds ZERO `NEXT` relationships
  with no error at runtime (see docs/chunk_schema.md).
- speech length -> `extraction_eligible`. A transcript is 15% silence markers,
  and the extraction prompt cannot answer "nothing here" — it requires 1-3
  Topics per chunk, so a chunk of `"you"` yields invented ones. See
  `DEFAULT_ELIGIBLE_MIN_CHARS`.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Transcript chunks are minutes of speech, so `source_kind` is not "pdf". This
# value also identifies the file as a meeting on its own if it ever comes back
# round through Supabase — see `infer_doc_type` in src/sync/supabase_sync.py.
SOURCE_KIND = "transcript"

# Below this many characters of actual speech, a chunk is marked
# `extraction_eligible: false` — kept and embedded, but never sent to the
# extraction LLM.
#
# 70 is measured, not guessed. Across the 57-file meeting corpus (2389 chunks)
# every one of the 372 chunks under it was read: 357 are the bare word "you"
# (what Whisper emits for silence), the rest are "Thank you.", "Bye bye.",
# "Thanks for watching!" (a Whisper training-data artifact). Zero carry content.
#
# Do NOT raise it to catch more. Just above the line sit real Action Items
# ("Tomorrow I will give two hours to talk to the Coral University." — 73 chars)
# and, at 235 chars, one of the densest passages in the corpus. Length stops
# being a usable signal the moment there is a real sentence: filler-line density
# is 0% in every band above 70, so nothing cheap separates the junk from the
# content up there. That band needs a judgement call, not a threshold.
DEFAULT_ELIGIBLE_MIN_CHARS = 70

# "[SPEAKER_02] " / "[unknown] " prefixes. Stripped before measuring, so the
# threshold counts speech rather than diarization scaffolding — with tags, a
# chunk of six "[SPEAKER_00] Okay." lines measures 100+ characters.
_SPEAKER_TAG = re.compile(r"^\[[^\]]*\]\s*", re.M)

# `<slug>-<YYYYMMDD>-<HHMMSS>[utc]-<...>` — every meeting_id in this corpus.
# Non-greedy so the earliest 8-digit date wins: "oral-defense-june-25-2026-
# 20260625-125907" keeps the spelled-out date in the slug and reads 20260625.
_MEETING_ID_RE = re.compile(r"^(?P<slug>.+?)-(?P<date>20\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))-\d{6}")


def speech_length(text: str) -> int:
    """Characters of actual speech in a chunk, ignoring the speaker tags."""
    return len(_SPEAKER_TAG.sub("", text or "").strip())


def parse_meeting_id(meeting_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    `(date, series)` read off a meeting_id, or `(None, None)` if it doesn't match.

    `date` is ISO 8601 (`"2026-01-16"`); `series` is the name with the timestamp
    and hash stripped (`"llm-bi-weekly-meeting"`). Both end up on the Document
    node, where they are the two properties `_create_precedes_relationships`
    needs to chain a recurring meeting into a timeline, and where `date` also
    becomes the `Source.year` the supersession pass requires.

    Read from the id rather than from the transcript on purpose. The extraction
    prompt is forbidden to guess a date that is not in the text, and a meeting's
    date is essentially never spoken aloud — but the producer knows it exactly
    and encodes it here, so this is authoritative metadata, not a guess.

    Series is name-derived and therefore only as good as the naming. Two known
    limits in this corpus: `rmfs-meeting-1023` and `rmfs-meeting-251113` carry a
    date inside the slug and so read as different series despite being the same
    recurring meeting; and `meeting-in-general` is a catch-all bucket, so its
    members are ordered in time but are not really one continuing series.
    """
    match = _MEETING_ID_RE.match(meeting_id or "")
    if not match:
        return None, None
    raw = match.group("date")
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}", match.group("slug")


# Producer fields worth keeping on the record. They do not reach Neo4j (the
# pipeline ignores unknown fields) but they survive in the .jsonl, so a chunk
# can still be traced back to its utterances and its point in the recording.
CARRIED_FIELDS = ("speaker_ids", "start_ms", "end_ms", "source_utterance_ids", "text_language")


def looks_like_meeting_payload(payload: Any) -> bool:
    """Whether a parsed JSON document is one of these files.

    Used to tell the two shapes apart on the sync path, where a meeting upload
    may be either this dialect or a `.jsonl` a producer already converted.
    """
    return isinstance(payload, dict) and isinstance(payload.get("chunks"), list)


def records_from_payload(
    payload: Dict[str, Any],
    eligible_min_chars: int = DEFAULT_ELIGIBLE_MIN_CHARS,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build ChunkRecord dicts from one parsed `*.chunks.json`. Returns
    `(records, notes)`; notes are things the caller should report, not failures.

    EVERY chunk becomes a record — a contentless one is marked
    `extraction_eligible: false`, never dropped. It still reaches Neo4j with its
    text, its embedding and its place in the NEXT chain (`store_chunks_for_doc`
    writes those before it ever looks at the extracted graph); only the LLM call
    is skipped. Dropping would have cost the transcript's real timeline and left
    `index` renumbered across the gaps for no gain.

    Pass `eligible_min_chars=0` to mark nothing — every chunk goes to the LLM.

    Raises `ValueError` when there is no `meeting_id` to group chunks by.
    """
    notes: List[str] = []
    meta = payload.get("metadata") or {}
    chunks = payload.get("chunks") or []

    declared = meta.get("conversation_chunks")
    if declared is not None and declared != len(chunks):
        notes.append(f"metadata.conversation_chunks={declared} but the file holds {len(chunks)} chunks")

    source_path = meta.get("input") or (chunks[0].get("source_uri") if chunks else None)

    records = []
    n_ineligible = 0
    for index, chunk in enumerate(chunks):
        doc_id = chunk.get("meeting_id") or meta.get("meeting_id")
        if not doc_id:
            raise ValueError("no meeting_id on the chunk or in metadata — nothing to group chunks by")

        text = chunk.get("text") or ""
        eligible = speech_length(text) >= eligible_min_chars
        n_ineligible += not eligible
        date, series = parse_meeting_id(doc_id)

        record: Dict[str, Any] = {
            "text": text,
            "doc_id": doc_id,
            # Array position, not the producer's string chunk_id — NEXT is built
            # with integer arithmetic in Cypher. Nothing is dropped above, so this
            # stays 1:1 with the transcript's real order.
            "index": index,
            "n_chunks": len(chunks),
            "source_path": source_path,
            "source_kind": SOURCE_KIND,
            "extraction_eligible": eligible,
            # Doc-level, read off the id — see parse_meeting_id. Repeated on
            # every record because ChunkRecord is per-chunk; the ingestor takes
            # them from the first record of each doc_id.
            "date": date,
            "series": series,
            # The producer's own id, kept for traceability. `index` above is what
            # the graph actually uses.
            "chunk_id": chunk.get("chunk_id"),
        }
        for field in CARRIED_FIELDS:
            if field in chunk:
                record[field] = chunk[field]
        records.append(record)

    if n_ineligible:
        notes.append(
            f"{n_ineligible}/{len(chunks)} chunk(s) under {eligible_min_chars} chars of speech "
            f"— stored, but not sent to the extraction LLM"
        )

    return records, notes


def to_lines(records: List[Dict[str, Any]]) -> List[str]:
    """The records as .jsonl lines — one compact JSON object each."""
    return [json.dumps(r, ensure_ascii=False) for r in records]
