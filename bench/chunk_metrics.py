"""
Intrinsic chunk-quality metrics — the CHEAP first pass, no LLM, no Neo4j, no
embeddings. Runs on the native chunk files directly and answers "which chunking
looks better suited to per-chunk KG extraction?" before you spend a single API
call.

Reads the NORMALIZED chunks produced by prep.py (so the same uniform cleaning /
empty-chunk filtering applies), and reports per method+doc:

  n_chunks            how many usable chunks the method produced
  chars mean/median   chunk size (too small -> relationships split across chunks
                      and lost by per-chunk extraction; too large -> LLM misses
                      relations, has_source logic degrades)
  chars p10/p90/max   size spread — a wildly uneven distribution hurts extraction
  tiny_chunks         chunks < TINY_CHARS: likely fragments that strand entities
  huge_chunks         chunks > HUGE_CHARS: likely to overflow / dilute extraction
  ends_sentence %     fraction ending on . ? ! 。 ？ ！ (proxy for clean boundaries;
                      mid-sentence cuts are where cross-chunk relations get lost)

Interpretation for KG (not RAG): the goal is chunks each holding a coherent unit
whose entities AND their relations sit together. Favor consistent, mid-sized,
sentence-aligned chunks over many tiny fragments.
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NORM_DIR = Path(__file__).resolve().parent / "normalized"

DEFAULT_DOCS = (
    "Thesis_M10801107_Yu_Ting_Chiu",
    "Thesis_M11001010_Hung_Chun_Tse_Nick",
)
METHODS = ("maruf", "wildan")

TINY_CHARS = 120
HUGE_CHARS = 3000
_SENT_END = tuple(".?!。？！\"')")


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def load_norm(method: str, doc: str):
    path = NORM_DIR / f"{method}__{doc}.jsonl"
    if not path.exists():
        return None
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def metrics_for(rows):
    lens = [len(r["text"]) for r in rows]
    if not lens:
        return None
    ends = sum(1 for r in rows if r["text"].rstrip().endswith(_SENT_END))
    return {
        "n": len(rows),
        "mean": st.mean(lens),
        "median": st.median(lens),
        "p10": sorted(lens)[max(0, int(0.10 * len(lens)) - 1)],
        "p90": sorted(lens)[min(len(lens) - 1, int(0.90 * len(lens)))],
        "max": max(lens),
        "tiny": sum(1 for x in lens if x < TINY_CHARS),
        "huge": sum(1 for x in lens if x > HUGE_CHARS),
        "ends_sentence_pct": pct(ends, len(rows)),
    }


def fmt_row(label, m):
    if m is None:
        return f"  {label:10} (no normalized file — run prep.py first)"
    return (f"  {label:10} n={m['n']:4}  "
            f"chars: mean={m['mean']:6.0f} med={m['median']:6.0f} "
            f"p10={m['p10']:5} p90={m['p90']:6} max={m['max']:6}  "
            f"tiny={m['tiny']:3} huge={m['huge']:2}  "
            f"ends_sent={m['ends_sentence_pct']:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", action="append", help="doc base name (repeatable).")
    args = ap.parse_args()
    docs = args.doc or list(DEFAULT_DOCS)

    for doc in docs:
        print(f"\n=== {doc} ===")
        for method in METHODS:
            rows = load_norm(method, doc)
            print(fmt_row(method, metrics_for(rows) if rows else None))
    print("\n(tiny < %d chars, huge > %d chars; ends_sent = clean sentence boundary)"
          % (TINY_CHARS, HUGE_CHARS))


if __name__ == "__main__":
    main()
