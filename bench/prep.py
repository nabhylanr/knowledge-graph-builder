"""
Normalize a document's chunks from EITHER chunking method (maruf / wildan) into
the flat .jsonl schema the pipeline loader (src/ingestion/chunks_ingestor.py)
expects: one JSON object per line with `text`, `doc_id`, `chunk_id`, `index`.

Why this exists: the two methods ship different native formats —
  - maruf : <doc>_<hash>.jsonl (flat, ready) AND <doc>_<hash>.json (nested)
  - wildan: <doc>.json only (nested under "chunks", no .jsonl at all)
so the loader can read maruf but NOT wildan. Normalizing both through one code
path makes the comparison apples-to-apples: same loader, same downstream
pipeline, only the chunking differs.

Fairness rules applied UNIFORMLY to both methods (so neither is handicapped):
  - a chunk's text is stripped of pure markup (e.g. wildan's "<!-- image -->"
    image placeholders, HTML comments);
  - a chunk is dropped only if what remains is shorter than MIN_CHARS — i.e. it
    carries no extractable content for the LLM. The count dropped is reported,
    because "how many empty/markup-only chunks did this method emit" is itself a
    chunking-quality signal.
  - method-specific flags (maruf's `extraction_eligible`) are IGNORED by default
    so both methods are treated identically; pass --respect-eligible to honor it.

Usage:
    python bench/prep.py                      # default 2 docs, both methods
    python bench/prep.py --doc Thesis_M10801107_Yu_Ting_Chiu
    python bench/prep.py --respect-eligible
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "chunks_data"
OUT_DIR = Path(__file__).resolve().parent / "normalized"

METHODS = ("maruf", "wildan")

# The 2 documents chosen for experiment #1 — present in BOTH maruf and wildan.
DEFAULT_DOCS = (
    "Thesis_M10801107_Yu_Ting_Chiu",
    "Thesis_M11001010_Hung_Chun_Tse_Nick",
)

MIN_CHARS = 15  # a chunk with less real text than this carries nothing to extract

_MARKUP = re.compile(r"<!--.*?-->|<[^>]+>", re.DOTALL)  # html comments + tags


def clean_text(raw: str) -> str:
    """Strip markup-only noise; return '' if nothing extractable remains."""
    if not raw:
        return ""
    t = _MARKUP.sub(" ", raw)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def find_source(method: str, doc: str) -> Path | None:
    """Locate a doc's native file in a method folder (filenames carry a hash suffix)."""
    folder = CHUNKS_DIR / method
    for ext in (".jsonl", ".json"):
        hits = sorted(folder.glob(f"{doc}*{ext}"))
        if hits:
            return hits[0]
    return None


def iter_native_chunks(path: Path):
    """Yield raw chunk dicts from a native file, regardless of method/format."""
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    # both maruf.json and wildan.json nest the list under "chunks"
    for c in data.get("chunks", []):
        yield c


def normalize_doc(method: str, doc: str, respect_eligible: bool) -> dict:
    src = find_source(method, doc)
    if src is None:
        return {"method": method, "doc": doc, "error": "source file not found"}

    kept, dropped_empty, dropped_ineligible = [], 0, 0
    for c in iter_native_chunks(src):
        if respect_eligible and c.get("extraction_eligible") is False:
            dropped_ineligible += 1
            continue
        text = clean_text(c.get("text", ""))
        if len(text) < MIN_CHARS:
            dropped_empty += 1
            continue
        # normalize index across schemas: maruf uses `index`, wildan `chunk_index`
        index = c.get("index", c.get("chunk_index", len(kept)))
        kept.append({
            "text": text,
            "doc_id": doc,
            "chunk_id": c.get("chunk_id", index),
            "index": index,
            "source_kind": "pdf",
            "method": method,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{method}__{doc}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "method": method,
        "doc": doc,
        "src": src.name,
        "kept": len(kept),
        "dropped_empty_or_markup": dropped_empty,
        "dropped_ineligible": dropped_ineligible,
        "out": str(out_path.relative_to(ROOT)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", action="append", help="doc base name (repeatable). Default: the 2 experiment docs.")
    ap.add_argument("--respect-eligible", action="store_true",
                    help="honor maruf's extraction_eligible=False flag (default: ignore, treat both methods identically).")
    args = ap.parse_args()

    docs = args.doc or list(DEFAULT_DOCS)
    print(f"Normalizing {len(docs)} doc(s) x {len(METHODS)} method(s) -> {OUT_DIR.relative_to(ROOT)}/\n")
    for doc in docs:
        for method in METHODS:
            r = normalize_doc(method, doc, args.respect_eligible)
            if "error" in r:
                print(f"  [{method:6}] {doc}: {r['error']}")
            else:
                print(f"  [{method:6}] {doc}: {r['kept']:3} chunks kept "
                      f"(dropped {r['dropped_empty_or_markup']} empty/markup"
                      + (f", {r['dropped_ineligible']} ineligible" if args.respect_eligible else "")
                      + f") -> {r['out']}")


if __name__ == "__main__":
    main()
