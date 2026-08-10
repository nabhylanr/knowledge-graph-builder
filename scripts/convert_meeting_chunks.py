"""
Convert a meeting chunker's `*.chunks.json` into the pipeline's chunks `.jsonl`.

    python scripts/convert_meeting_chunks.py path/to/meeting.chunks.json
    python scripts/convert_meeting_chunks.py transcripts/ --min-chars 60
    python scripts/convert_meeting_chunks.py meeting.chunks.json --dry-run

The meeting producer emits ONE JSON object holding a `chunks` array
(`input_schema_version: speaker_transcript.v1`), which the ingestor cannot read.
The translation lives in `src.ingestion.meeting_chunks`; this script is only the
command-line front end for files that are already on disk. Files arriving
through Supabase go through the same adapter inside `run_sync.py`, so both routes
produce byte-identical .jsonl.

Output goes to `chunks_data/meeting/<doc_id>.jsonl` so the build ledger tags the
document `meeting` (the doc_type is the parent folder, see
`src.ingestion.build_ledger.doc_type_of`). Every file is validated before it is
written; a file with errors is refused.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Run directly as `python scripts/convert_meeting_chunks.py` — sys.path[0] is
# then this script's folder, not the repo root, so `src` needs putting on it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.meeting_chunks import (
    DEFAULT_ELIGIBLE_MIN_CHARS,
    looks_like_meeting_payload,
    records_from_payload,
    to_lines,
)
from src.ingestion.validate import print_report, validate_lines

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "chunks_data" / "meeting"


def convert(path: Path, out_dir: Path, eligible_min_chars: int, dry_run: bool) -> bool:
    """Convert one file. Returns True if it is clean (and written, unless dry-run)."""

    def fail(message: str) -> bool:
        # print_report prints the path itself, so a failure before it has to.
        print(f"\n{path}\n  ERROR: {message}")
        return False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return fail(f"cannot read as JSON ({e})")

    if not looks_like_meeting_payload(payload):
        return fail("not a meeting chunks file (no top-level 'chunks' array)")

    try:
        records, notes = records_from_payload(payload, eligible_min_chars=eligible_min_chars)
    except ValueError as e:
        return fail(str(e))

    if not records:
        return fail("no chunks left to write")

    lines = to_lines(records)
    report = validate_lines(lines, path=str(path))
    print_report(report, verbose=False)
    for note in notes:
        print(f"  note: {note}")
    if not report.ok:
        print("  refusing to write a file the pipeline would reject")
        return False

    dest = out_dir / f"{records[0]['doc_id']}.jsonl"
    if dry_run:
        print(f"  would write {dest} ({len(records)} records)")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {dest} ({len(records)} records)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert meeting *.chunks.json files to the pipeline's chunks .jsonl.")
    parser.add_argument("paths", nargs="+", help="A *.chunks.json file, or a folder searched recursively.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output folder (default: {DEFAULT_OUT}).")
    parser.add_argument(
        "--eligible-min-chars",
        type=int,
        default=DEFAULT_ELIGIBLE_MIN_CHARS,
        help=f"Mark chunks with less than this many characters of speech as "
             f"extraction_eligible:false — they are still written and still reach Neo4j with their "
             f"embedding, they just never reach the extraction LLM (default: {DEFAULT_ELIGIBLE_MIN_CHARS}). "
             f"0 marks nothing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report, write nothing.")
    args = parser.parse_args()

    files: List[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.chunks.json")) or sorted(path.rglob("*.json")))
        else:
            files.append(path)

    if not files:
        print("No files found.")
        sys.exit(1)

    ok = [convert(p, Path(args.out), args.eligible_min_chars, args.dry_run) for p in files]
    print(f"\n{len(files)} file(s): {sum(ok)} converted, {len(ok) - sum(ok)} failed")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
