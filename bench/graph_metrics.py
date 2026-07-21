"""
Graph-quality metrics — the REAL comparison. Runs each method's normalized chunks
through the actual extraction path (GraphExtractor + sanitize_graph, the SAME code
the pipeline uses) and measures the graph that comes out. No Neo4j and no
embeddings needed: benchmarking chunking only needs the LLM extraction, so this
builds the graph in-memory with networkx. Only requirement is the RE_MODEL_* /
Groq credentials in .env (same as `python main.py`).

For each method+doc it reports:

  raw_nodes         nodes the LLM emitted before sanitizing (raw verbosity)
  nodes / rels      unique nodes / relationships AFTER sanitize + cross-chunk merge
  dup_ratio         sanitized-nodes-emitted / unique-nodes. ~1.0 = each entity
                    appears once; HIGH = the same entity is re-emitted across many
                    chunks, i.e. the chunking split entities apart and the pipeline
                    had to merge them back. Lower is better.
  lcc_frac          largest connected component / all nodes. HIGH = one coherent
                    graph; LOW = many disconnected islands (relations were lost at
                    chunk boundaries). Primary signal — higher is better.
  isolated%         nodes with zero edges (extraction that linked to nothing).
  degree1%          nodes with exactly one edge (weakly attached). Lower is better.
  avg_degree        mean edges per node.
  louvain/leiden    modularity + community count. Read TOGETHER with lcc_frac:
                    high modularity + low lcc = fragmented, not good structure.

Writes the extracted graph to bench/out/<method>__<doc>.graph.json so eval_gold.py
can score it against the gold set without re-calling the LLM.

Usage:
    python bench/graph_metrics.py                 # both docs, both methods
    python bench/graph_metrics.py --doc Thesis_M10801107_Yu_Ting_Chiu
    python bench/graph_metrics.py --limit 20      # first N chunks only (quick smoke)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.graph_extractor import GraphExtractor              # noqa: E402
from src.config import LLMConf                                     # noqa: E402
from src.graph.graph_model import sanitize_graph, _canonical_id    # noqa: E402

# Modularity needs python-louvain + leidenalg + igraph (C extensions). Optional:
# if they aren't installed, the core structural metrics (lcc, dup, isolated,
# degree) still compute and modularity is reported as n/a.
try:
    from src.graph.graph_ds import detect_louvain_communities, detect_leiden_communities
    _HAVE_COMMUNITY = True
except Exception:
    _HAVE_COMMUNITY = False

NORM_DIR = Path(__file__).resolve().parent / "normalized"
OUT_DIR = Path(__file__).resolve().parent / "out"

DEFAULT_DOCS = (
    "Thesis_M10801107_Yu_Ting_Chiu",
    "Thesis_M11001010_Hung_Chun_Tse_Nick",
)
METHODS = ("maruf", "wildan")


def build_llm_conf() -> LLMConf:
    load_dotenv(ROOT / ".env")
    return LLMConf(
        type=os.getenv("RE_MODEL_TYPE", "groq"),
        model=os.getenv("RE_MODEL_NAME"),
        temperature=os.getenv("RE_MODEL_TEMPERATURE") or 0.0,
        deployment=os.getenv("RE_MODEL_DEPLOYMENT"),
        api_key=os.getenv("RE_API_KEY"),
        endpoint=os.getenv("RE_MODEL_ENDPOINT"),
        api_version=os.getenv("RE_MODEL_API_VERSION") or None,
    )


def load_chunks(method: str, doc: str):
    path = NORM_DIR / f"{method}__{doc}.jsonl"
    if not path.exists():
        return None
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def mine_method(extractor: GraphExtractor, chunks, doc: str):
    """
    Replicates GraphMiner's per-chunk loop but keeps the RAW graph too, so we can
    measure how much the chunking made the pipeline merge. Accumulates a global,
    canonical-id-keyed node/edge set == the graph that would land in Neo4j.
    """
    has_source_state, topic_registry = {}, {}
    raw_nodes = 0
    sanitized_nodes_emitted = 0          # summed across chunks, pre global-merge
    nodes = {}                           # canonical id -> type  (global merge)
    edges = {}                           # (src, tgt, type) -> properties

    for i, ch in enumerate(chunks):
        try:
            g = extractor.extract_graph(text=ch["text"], source_name=doc, source_format="pdf")
        except Exception as e:
            print(f"    ! chunk {i}: extract error {e}", file=sys.stderr)
            continue
        if g is None:
            continue
        raw_nodes += len(g.nodes)
        sg = sanitize_graph(g, source_name=doc,
                            has_source_state=has_source_state, topic_registry=topic_registry)
        if sg is None:
            continue
        sanitized_nodes_emitted += len(sg.nodes)
        for n in sg.nodes:
            nodes[_canonical_id(n.id)] = n.type
        for r in sg.relationships:
            edges[(_canonical_id(r.source), _canonical_id(r.target), r.type)] = r.properties or {}

    return raw_nodes, sanitized_nodes_emitted, nodes, edges


def graph_stats(nodes: dict, edges: dict):
    G = nx.DiGraph()
    for nid, ntype in nodes.items():
        G.add_node(nid, type=ntype)
    for (s, t, rt), props in edges.items():
        G.add_edge(s, t, type=rt, **{k: v for k, v in props.items() if isinstance(v, (str, int, float))})

    n = G.number_of_nodes()
    if n == 0:
        return None, G

    UG = G.to_undirected()
    comps = list(nx.connected_components(UG))
    lcc = max((len(c) for c in comps), default=0)
    isolated = sum(1 for _, d in UG.degree() if d == 0)
    degree1 = sum(1 for _, d in UG.degree() if d == 1)
    avg_degree = sum(d for _, d in G.degree()) / n

    louv_mod = leid_mod = None
    n_louv = n_leid = None
    if _HAVE_COMMUNITY:
        try:
            _, louv_mod = detect_louvain_communities(G, return_modularity=True)
            n_louv = len({data.get("community_louvain") for _, data in _.nodes(data=True)
                          if data.get("community_louvain") is not None})
        except Exception:
            pass
        try:
            _, leid_mod = detect_leiden_communities(G, return_modularity=True)
            n_leid = len({data.get("community_leiden") for _, data in _.nodes(data=True)
                          if data.get("community_leiden") is not None})
        except Exception:
            pass

    return {
        "nodes": n,
        "rels": G.number_of_edges(),
        "n_components": len(comps),
        "lcc_frac": lcc / n,
        "isolated_pct": 100.0 * isolated / n,
        "degree1_pct": 100.0 * degree1 / n,
        "avg_degree": avg_degree,
        "louvain_mod": louv_mod,
        "n_louvain": n_louv,
        "leiden_mod": leid_mod,
        "n_leiden": n_leid,
    }, G


def dump_graph(method: str, doc: str, nodes: dict, edges: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "method": method,
        "doc": doc,
        "nodes": [{"id": nid, "type": t} for nid, t in nodes.items()],
        "relationships": [
            {"source": s, "target": t, "type": rt, "properties": props}
            for (s, t, rt), props in edges.items()
        ],
    }
    path = OUT_DIR / f"{method}__{doc}.graph.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def fmt(method, raw_nodes, emitted, stats):
    if stats is None:
        return f"  {method:8} (empty graph)"
    dup = emitted / stats["nodes"] if stats["nodes"] else 0.0
    return (f"  {method:8} raw={raw_nodes:4}  nodes={stats['nodes']:4} rels={stats['rels']:4}  "
            f"dup={dup:4.2f}  lcc={stats['lcc_frac']*100:5.1f}%  "
            f"iso={stats['isolated_pct']:4.1f}%  deg1={stats['degree1_pct']:4.1f}%  "
            f"avgdeg={stats['avg_degree']:4.2f}  "
            f"louv_mod={_f(stats['louvain_mod'])}/{stats['n_louvain']}  "
            f"leid_mod={_f(stats['leiden_mod'])}/{stats['n_leiden']}")


def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else " n/a "


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", action="append", help="doc base name (repeatable).")
    ap.add_argument("--limit", type=int, default=0, help="only first N chunks per method (quick smoke).")
    args = ap.parse_args()
    docs = args.doc or list(DEFAULT_DOCS)

    conf = build_llm_conf()
    if not conf.model or not conf.api_key:
        print("ERROR: RE_MODEL_NAME / RE_API_KEY not set in .env — extraction needs the LLM.", file=sys.stderr)
        sys.exit(1)
    extractor = GraphExtractor(conf=conf)

    for doc in docs:
        print(f"\n=== {doc} ===")
        for method in METHODS:
            chunks = load_chunks(method, doc)
            if chunks is None:
                print(f"  {method:8} (no normalized file — run prep.py first)")
                continue
            if args.limit:
                chunks = chunks[:args.limit]
            print(f"  {method:8} mining {len(chunks)} chunks...", file=sys.stderr)
            raw_nodes, emitted, nodes, edges = mine_method(extractor, chunks, doc)
            stats, _ = graph_stats(nodes, edges)
            dump_graph(method, doc, nodes, edges)
            print(fmt(method, raw_nodes, emitted, stats))
    print("\n(dup~1=no cross-chunk duplication; lcc high=connected; iso/deg1 low=well-linked)")


if __name__ == "__main__":
    main()
