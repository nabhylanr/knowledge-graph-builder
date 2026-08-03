"""
Validate a chunks `.jsonl` file against the ChunkRecord contract.

Run this BEFORE handing a file to the pipeline — a bad file is otherwise only
noticed hours into extraction, or not at all (a missing `index` produces a
perfectly quiet graph with no NEXT chain).

    python -m src.ingestion.validate chunks_data/maruf
    python -m src.ingestion.validate path/to/one_file.jsonl

Exits 1 if any file has errors, so it drops straight into a CI step or the
sync job. Depends only on pydantic — a chunk producer can run it on their own
machine without installing the rest of the pipeline.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from pydantic import ValidationError

from src.ingestion.chunk_record import ChunkRecord

# Chunks shorter than this are dropped by the conflict-detection gate later
# (GATE_MIN_TEXT_LENGTH). Not fatal, but worth surfacing as a count.
SHORT_TEXT_CHARS = 40

# How many individual issues of one kind to print before collapsing the rest.
MAX_LISTED = 8


@dataclass
class DocSummary:
    """Per-document accumulation, checked once the whole file is read."""

    doc_id: str
    indices: List[int] = field(default_factory=list)
    n_missing_index: int = 0
    declared_n_chunks: Optional[int] = None
    source_kind: Optional[str] = None
    n_records: int = 0


@dataclass
class FileReport:
    path: str
    n_records: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    docs: Dict[str, DocSummary] = field(default_factory=dict)
    unknown_fields: Set[str] = field(default_factory=set)
    n_ineligible: int = 0
    n_short_text: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<record>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def validate_lines(lines: Iterable[str], path: str = "<stdin>") -> FileReport:
    report = FileReport(path=path)
    seen: Set[Tuple[str, Union[int, str, None]]] = set()

    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            report.errors.append(f"line {lineno}: invalid JSON ({e.msg})")
            continue

        if not isinstance(raw, dict):
            report.errors.append(f"line {lineno}: expected a JSON object, got {type(raw).__name__}")
            continue

        try:
            record = ChunkRecord.model_validate(raw)
        except ValidationError as e:
            report.errors.append(f"line {lineno}: {_format_validation_error(e)}")
            continue

        report.n_records += 1
        report.unknown_fields.update(record.unknown_fields)

        doc_id = record.effective_doc_id
        if doc_id == "unknown":
            report.errors.append(
                f"line {lineno}: no doc_id and no source_path — chunks cannot be grouped into a document"
            )

        key = (doc_id, record.effective_chunk_id)
        if key in seen:
            report.errors.append(
                f"line {lineno}: duplicate chunk id {record.effective_chunk_id!r} in doc {doc_id!r} "
                "— the second one would overwrite the first in the graph"
            )
        seen.add(key)

        doc = report.docs.setdefault(doc_id, DocSummary(doc_id=doc_id))
        doc.n_records += 1
        if record.index is None:
            doc.n_missing_index += 1
        else:
            doc.indices.append(record.index)
        if doc.declared_n_chunks is None:
            doc.declared_n_chunks = record.n_chunks
        if doc.source_kind is None:
            doc.source_kind = record.source_kind

        if not record.text.strip():
            report.warnings.append(f"line {lineno}: text is empty — the chunk yields nothing to extract")
        elif len(record.text.strip()) < SHORT_TEXT_CHARS:
            report.n_short_text += 1

        if not record.extraction_eligible:
            report.n_ineligible += 1

    _check_docs(report)
    return report


def _check_docs(report: FileReport) -> None:
    """Whole-document checks that can only run once every line has been read."""
    for doc in report.docs.values():
        if doc.n_missing_index:
            report.warnings.append(
                f"doc {doc.doc_id!r}: {doc.n_missing_index}/{doc.n_records} records have no integer `index` "
                "— NEXT relationships will not be built for them"
            )

        if doc.indices:
            expected = set(range(min(doc.indices), max(doc.indices) + 1))
            missing = sorted(expected - set(doc.indices))
            if missing:
                shown = ", ".join(str(i) for i in missing[:MAX_LISTED])
                more = f" (+{len(missing) - MAX_LISTED} more)" if len(missing) > MAX_LISTED else ""
                report.warnings.append(
                    f"doc {doc.doc_id!r}: gaps in `index` at {shown}{more} — the NEXT chain breaks there"
                )
            if min(doc.indices) != 0:
                report.warnings.append(
                    f"doc {doc.doc_id!r}: `index` starts at {min(doc.indices)}, not 0"
                )

        if doc.declared_n_chunks is not None and doc.declared_n_chunks != doc.n_records:
            report.warnings.append(
                f"doc {doc.doc_id!r}: declares n_chunks={doc.declared_n_chunks} "
                f"but the file holds {doc.n_records} records"
            )

        if not doc.source_kind:
            report.warnings.append(
                f"doc {doc.doc_id!r}: no source_kind/source_type — Source nodes will be labelled format 'unknown'"
            )


def validate_file(path: Path) -> FileReport:
    with open(path, "r", encoding="utf-8") as f:
        return validate_lines(f, path=str(path))


def collect_paths(target: Path) -> List[Path]:
    if target.is_dir():
        return sorted(target.rglob("*.jsonl"))
    return [target]


def print_report(report: FileReport, verbose: bool = False) -> None:
    total_chunks = sum(d.n_records for d in report.docs.values())
    print(f"\n{report.path}")
    print(f"  {report.n_records} records · {len(report.docs)} document(s) · {total_chunks} chunks")

    for doc in report.docs.values():
        bits = [f"source_kind={doc.source_kind or '—'}"]
        if doc.declared_n_chunks is not None:
            bits.append(f"n_chunks={doc.declared_n_chunks}")
        print(f"    {doc.doc_id}  ({', '.join(bits)})")

    if report.unknown_fields:
        print(f"  fields carried but unused by the pipeline: {', '.join(sorted(report.unknown_fields))}")
    if report.n_ineligible:
        print(f"  extraction_eligible=false: {report.n_ineligible} chunk(s)")
    if report.n_short_text:
        print(f"  shorter than {SHORT_TEXT_CHARS} chars: {report.n_short_text} chunk(s) (dropped later by the conflict gate)")

    for label, issues in (("ERROR", report.errors), ("WARN", report.warnings)):
        limit = len(issues) if verbose else MAX_LISTED
        for issue in issues[:limit]:
            print(f"  {label}: {issue}")
        if len(issues) > limit:
            print(f"  {label}: ... and {len(issues) - limit} more (use --verbose)")

    if report.ok and not report.warnings:
        print("  OK")
    elif report.ok:
        print(f"  OK with {len(report.warnings)} warning(s)")
    else:
        print(f"  FAILED — {len(report.errors)} error(s)")


def run(targets: List[str], verbose: bool = False) -> int:
    paths: List[Path] = []
    for target in targets:
        path = Path(target)
        if not path.exists():
            print(f"{target}: no such file or directory", file=sys.stderr)
            return 2
        paths.extend(collect_paths(path))

    if not paths:
        print("No .jsonl files found.", file=sys.stderr)
        return 2

    reports = [validate_file(p) for p in paths]
    for report in reports:
        print_report(report, verbose=verbose)

    n_failed = sum(1 for r in reports if not r.ok)
    n_warned = sum(1 for r in reports if r.ok and r.warnings)
    print(
        f"\n{len(reports)} file(s): {len(reports) - n_failed - n_warned} clean, "
        f"{n_warned} with warnings, {n_failed} failed"
    )
    return 1 if n_failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate chunks .jsonl files against the ChunkRecord contract (see docs/chunk_schema.md)."
    )
    parser.add_argument("paths", nargs="+", help="A .jsonl file, or a folder searched recursively.")
    parser.add_argument("--verbose", action="store_true", help="List every issue instead of the first few.")
    args = parser.parse_args()
    sys.exit(run(args.paths, verbose=args.verbose))


if __name__ == "__main__":
    main()
