"""
Producer-side uploader — this is the script Maruf runs on his machine.

    python scripts/upload_chunks.py out/*.jsonl
    python scripts/upload_chunks.py out/ --dry-run     # validate only, upload nothing

Validates every file against docs/chunk_schema.md BEFORE uploading, and refuses
to upload one with errors. That is the whole point of the exercise: a bad file
should fail here, in two seconds, on the machine of the person who can fix it —
not five hours into an extraction run on someone else's laptop.

Needs only `pip install supabase pydantic` plus this repo's
`src/ingestion/` — not the rest of the pipeline.

Credentials come from the environment (or a .env next to this repo):

    SUPABASE_URL=https://<project>.supabase.co
    SUPABASE_ANON_KEY=<the anon/public key>
    SUPABASE_EMAIL=<the account created for you>
    SUPABASE_PASSWORD=<...>
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Run directly as `python scripts/upload_chunks.py` — sys.path[0] is then this
# script's folder, not the repo root, so `src` has to be put on the path by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.validate import collect_paths, print_report, validate_lines

BUCKET = os.getenv("SUPABASE_BUCKET", "chunks")
TABLE = "chunk_uploads"


def _doc_id_of(content: bytes) -> Optional[str]:
    """doc_id from the first non-empty record — the file is already validated,
    so every record agrees on it."""
    for line in content.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            record = json.loads(line)
            return record.get("doc_id") or Path(record.get("source_path") or record.get("source_file", "unknown")).stem
    return None


def _sign_in():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    email = os.getenv("SUPABASE_EMAIL")
    password = os.getenv("SUPABASE_PASSWORD")
    missing = [n for n, v in [
        ("SUPABASE_URL", url), ("SUPABASE_ANON_KEY", key),
        ("SUPABASE_EMAIL", email), ("SUPABASE_PASSWORD", password),
    ] if not v]
    if missing:
        sys.exit(f"Missing environment variable(s): {', '.join(missing)}")

    client = create_client(url, key)
    client.auth.sign_in_with_password({"email": email, "password": password})
    return client, email


def upload_one(client, prefix: str, path: Path, content: bytes, n_records: int) -> str:
    doc_id = _doc_id_of(content)
    if not doc_id:
        return f"SKIP {path.name}: file is empty"

    digest = hashlib.sha256(content).hexdigest()
    storage_path = f"{prefix}/{doc_id}__{digest[:8]}.jsonl"

    try:
        client.storage.from_(BUCKET).upload(
            storage_path, content, {"content-type": "application/x-ndjson"}
        )
    except Exception as e:
        # A duplicate object means these exact bytes are already up there; the
        # manifest insert below is what actually decides whether it is new work.
        if "Duplicate" not in str(e) and "already exists" not in str(e):
            return f"FAIL {path.name}: upload failed: {e}"

    try:
        client.table(TABLE).insert({
            "doc_id": doc_id,
            "storage_path": storage_path,
            "sha256": digest,
            "n_chunks": n_records,
        }).execute()
    except Exception as e:
        if "duplicate key" in str(e).lower() or "23505" in str(e):
            return f"SKIP {path.name}: already uploaded (identical content)"
        return f"FAIL {path.name}: could not register upload: {e}"

    return f"OK   {path.name} -> {storage_path} ({n_records} chunks)"


def run(targets: List[str], prefix: Optional[str], dry_run: bool, verbose: bool) -> int:
    paths: List[Path] = []
    for target in targets:
        p = Path(target)
        if not p.exists():
            sys.exit(f"{target}: no such file or directory")
        paths.extend(collect_paths(p))

    if not paths:
        sys.exit("No .jsonl files found.")

    # Validate everything first — one bad file stops the whole batch, so a
    # half-uploaded set never has to be reasoned about.
    checked = []
    n_failed = 0
    for path in paths:
        content = path.read_bytes()
        report = validate_lines(content.decode("utf-8", errors="replace").splitlines(), path=str(path))
        print_report(report, verbose=verbose)
        if not report.ok:
            n_failed += 1
        checked.append((path, content, report.n_records))

    if n_failed:
        print(f"\n{n_failed} file(s) failed validation — nothing uploaded. Fix them and re-run.")
        return 1

    if dry_run:
        print(f"\nDry run: {len(checked)} file(s) valid, nothing uploaded.")
        return 0

    client, email = _sign_in()
    resolved_prefix = prefix or email.split("@")[0]
    print(f"\nUploading as {email} into {BUCKET}/{resolved_prefix}/ ...")

    results = [upload_one(client, resolved_prefix, path, content, n) for path, content, n in checked]
    for line in results:
        print(f"  {line}")

    n_ok = sum(1 for r in results if r.startswith("OK"))
    n_skipped = sum(1 for r in results if r.startswith("SKIP"))
    n_bad = sum(1 for r in results if r.startswith("FAIL"))
    print(f"\n{n_ok} uploaded, {n_skipped} skipped, {n_bad} failed.")
    return 1 if n_bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and upload chunk .jsonl files to Supabase.")
    parser.add_argument("paths", nargs="+", help="A .jsonl file, or a folder searched recursively.")
    parser.add_argument("--as", dest="prefix", default=None, help="Folder prefix in the bucket (default: your email's local part).")
    parser.add_argument("--dry-run", action="store_true", help="Validate only — do not upload.")
    parser.add_argument("--verbose", action="store_true", help="List every validation issue.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional on the producer side

    sys.exit(run(args.paths, prefix=args.prefix, dry_run=args.dry_run, verbose=args.verbose))


if __name__ == "__main__":
    main()
