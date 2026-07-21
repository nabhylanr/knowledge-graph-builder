"""
Gold-set evaluation — scores each method's EXTRACTED graph (written by
graph_metrics.py to bench/out/) against the human-curated gold set in bench/gold/.

This is the task-based signal: "did the chunking preserve enough context for the
LLM to recover the entities/relations we KNOW are in this document?" A chunking
that splits an entity's context across chunk boundaries loses it -> lower recall.

We report RECALL only (matched_gold / total_gold), for topics, agents, and
relations. Precision is deliberately NOT computed: the gold set is built from the
abstract and is not exhaustive, so a method extracting real body-text entities
absent from the gold set would be unfairly penalized. Pair this recall with the
graph-structure metrics from graph_metrics.py for the full picture.

Matching is canonicalized (same _canonical_id as the pipeline) with per-entity
aliases; short/abbreviation aliases match exactly, multi-word aliases match on a
whole-phrase word boundary so "Robotic Mobile Fulfillment System" is found inside
the abbreviation-merged node "Robotic Mobile Fulfillment System Rmfs".

Usage:
    python bench/eval_gold.py            # both docs, both methods (needs bench/out/*.graph.json)
    python bench/eval_gold.py --doc Thesis_M10801107_Yu_Ting_Chiu
    python bench/eval_gold.py -v         # also list what each method missed
"""
import argparse
import json
import re
from pathlib import Path


# Mirrors src/graph/graph_model._normalize / _canonical_id EXACTLY. Inlined here
# (not imported) so this recall check runs on plain JSON without pulling in the
# langchain/networkx stack the pipeline needs. Keep in sync with graph_model.py.
def _normalize(s: str) -> str:
    return re.sub(r'[.,;:!?@#$%^&*()\-_\[\]{}<>/\\\'"~\s]', ' ', s)


def _canonical_id(s: str) -> str:
    return re.sub(r"\s+", " ", _normalize(s)).strip().title()

GOLD_DIR = Path(__file__).resolve().parent / "gold"
OUT_DIR = Path(__file__).resolve().parent / "out"
METHODS = ("maruf", "wildan")
DEFAULT_DOCS = (
    "Thesis_M10801107_Yu_Ting_Chiu",
    "Thesis_M11001010_Hung_Chun_Tse_Nick",
)


def alias_matches(alias: str, node_canon: str) -> bool:
    a = _canonical_id(alias)
    if not a:
        return False
    if a == node_canon:
        return True
    # multi-word or long aliases: whole-phrase, word-boundary containment
    if len(a.split()) >= 2 or len(a) >= 8:
        return f" {a} " in f" {node_canon} "
    return False


def entity_found(gold_entity: dict, node_canons: list):
    """Return the matched node id, or None."""
    aliases = [gold_entity["canonical"]] + gold_entity.get("aliases", [])
    for nc in node_canons:
        if any(alias_matches(al, nc) for al in aliases):
            return nc
    return None


def eval_entities(gold_list, node_canons):
    hits, misses = [], []
    for g in gold_list:
        m = entity_found(g, node_canons)
        (hits if m else misses).append((g["canonical"], m))
    recall = len(hits) / len(gold_list) if gold_list else 1.0
    return recall, hits, misses


def eval_relations(gold_rels, edges, node_canons):
    """edges: list of (src_canon, tgt_canon, relation_or_type). Direction-agnostic."""
    found = 0
    detail = []
    for gr in gold_rels:
        src_ok = {n for n in node_canons
                  if any(alias_matches(s, n) for s in gr["source_any"])}
        tgt_ok = {n for n in node_canons
                  if any(alias_matches(t, n) for t in gr["target_any"])}
        want = gr["relation"].lower()
        hit = any(
            ((s in src_ok and t in tgt_ok) or (t in src_ok and s in tgt_ok)) and rel == want
            for (s, t, rel) in edges
        )
        found += int(hit)
        detail.append((f"{'/'.join(gr['source_any'])} --{want}--> {'/'.join(gr['target_any'])}", hit))
    recall = found / len(gold_rels) if gold_rels else 1.0
    return recall, detail


def load_graph(method, doc):
    p = OUT_DIR / f"{method}__{doc}.graph.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", action="append")
    ap.add_argument("-v", "--verbose", action="store_true", help="list misses")
    args = ap.parse_args()
    docs = args.doc or list(DEFAULT_DOCS)

    for doc in docs:
        gold_path = GOLD_DIR / f"{doc}.gold.json"
        if not gold_path.exists():
            print(f"\n=== {doc} === (no gold file)")
            continue
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        print(f"\n=== {doc} ===")
        print(f"    gold: {len(gold['topics'])} topics, {len(gold['agents'])} agents, "
              f"{len(gold.get('relations', []))} relations   [{gold['status']}]")
        for method in METHODS:
            g = load_graph(method, doc)
            if g is None:
                print(f"  {method:8} (no bench/out graph — run graph_metrics.py first)")
                continue
            node_canons = [_canonical_id(n["id"]) for n in g["nodes"]]
            edges = [(_canonical_id(r["source"]), _canonical_id(r["target"]),
                      (r.get("properties", {}) or {}).get("relation", r["type"]).lower())
                     for r in g["relationships"]]

            t_rec, t_hits, t_miss = eval_entities(gold["topics"], node_canons)
            a_rec, a_hits, a_miss = eval_entities(gold["agents"], node_canons)
            r_rec, r_detail = eval_relations(gold.get("relations", []), edges, node_canons)

            print(f"  {method:8} topic_recall={t_rec*100:5.1f}%  agent_recall={a_rec*100:5.1f}%  "
                  f"rel_recall={r_rec*100:5.1f}%   (nodes={len(g['nodes'])})")
            if args.verbose:
                if t_miss:
                    print("             missed topics: " + ", ".join(c for c, _ in t_miss))
                if a_miss:
                    print("             missed agents: " + ", ".join(c for c, _ in a_miss))
                for label, hit in r_detail:
                    if not hit:
                        print(f"             missed rel:   {label}")
    print("\n(recall only — see module docstring for why precision is not scored)")


if __name__ == "__main__":
    main()
