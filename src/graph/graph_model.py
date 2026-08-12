import difflib
import re
import networkx as nx

from typing import List, Dict, Any, Optional, Set, Tuple

from langchain.schema import Document
from langchain.load.serializable import Serializable
from langchain_neo4j.graphs.graph_document import Node, Relationship, GraphDocument

from src.utils.logger import get_logger


logger = get_logger(__name__)


class _Node(Serializable):
    id: str
    type: str
    properties: Optional[Dict[str, str]] = None


class _Relationship(Serializable):
    source: str
    target: str
    type: str
    properties: Optional[Dict[str, str]] = None


class _Graph(Serializable):
    """ 
    Represents a graph consisting of nodes and relationships.  
    
    -----------
    Attributes:
    -----------
        `nodes (List[_Node])`: A list of nodes in the graph.
        `relationships (List[_Relationship])`: A list of relationships in the graph.
    """
    nodes: List[_Node]
    relationships: List[_Relationship]


def graph_document_to_digraph(graph_doc: GraphDocument) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in graph_doc.nodes:
        G.add_node(
            node.id, 
            type=node.type
        )
    for relationship in graph_doc.relationships:
        G.add_edge(
            relationship.source.id, 
            relationship.target.id, 
            relationship=relationship.type, 
        )
    return G


def digraph_to_dict(G: nx.DiGraph, remove_unknown: bool=True) -> dict:

    graph_dict = {}
    
    for node in G.nodes(data=True):
        node_id = node[0]
        node_type = node[1]['type'] if 'type' in node[1].keys() else "unknown"
        graph_dict[node_id] = {'type': node_type, 'relationships': []}
        
    for node_id in G.nodes():
        successors = [
            (successor, G[node_id][successor].get('relationship', 'unknown')) 
            for successor in G.successors(node_id)
        ]        
        graph_dict[node_id]['relationships'] = successors
    
    if remove_unknown:
        graph_dict = remove_unknown_relationships(document_graph=graph_dict)
        
    return graph_dict


def dict_to_graph_document(graph_dict: Dict[str, Any], source_content: str="") -> GraphDocument:
    
    nodes = []
    nodes_map = {}  # To map node IDs to Node objects
    for node_id, node_info in graph_dict.items():
        node = Node(id=node_id, type=node_info['type'])
        nodes.append(node)
        nodes_map[node_id] = node
    
    relationships = []
    for node_id, node_info in graph_dict.items():
        for successor, relationship_type in node_info['relationships']:
            relationship = Relationship(
                source=nodes_map[node_id],
                target=nodes_map[successor],
                type=relationship_type
            )
            relationships.append(relationship)
    
    source = Document(page_content=source_content)
    
    graph_doc = GraphDocument(
        nodes=nodes, 
        relationships=relationships, 
        source=source
    )
    
    return graph_doc


def remove_unknown_relationships(document_graph: dict) -> dict:
    for key, value in document_graph.items():
        if 'relationships' in value:
            value['relationships'] = [
                relationship for relationship in value['relationships']
                if 'unknown' not in relationship
            ]
    return document_graph


def normalize_nodes(G: nx.DiGraph) -> nx.DiGraph:
    """Normalize Nodes names"""
    mapping = {node: _normalize(node) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G
    

def _normalize(s: str) -> str:
    return re.sub(r'[.,;:!?@#$%^&*()\-_\[\]{}<>/\\\'"~\s]', ' ', s)


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
      return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties


ALLOWED_LABELS = {"Agent", "Role", "Topic", "Type", "Source", "Description", "Contradiction"}


def _canonical_id(s: str) -> str:
    """
    Canonical form of a node id so that casing / underscore / spacing variants
    collapse into a single node on MERGE (e.g. "Research_Problem" and
    "Research Problem" both become "Research Problem").
    """
    return re.sub(r"\s+", " ", _normalize(s)).strip().title()


_FIXED_RELATION_DIRS = {
    "role_in_meeting": ("Agent", "Role"),
    "role_in_paper": ("Agent", "Role"),
    "spoke_about": ("Agent", "Topic"),
    "writes_about": ("Agent", "Topic"),
    "has_source": ("Topic", "Source"),
    "has_subtopic": ("Topic", "Topic"),
    "relates_to": ("Topic", "Topic"),
    "assigned_to": ("Topic", "Agent"),
    "has_contradiction": ("Description", "Contradiction"),
}

MAX_HAS_SOURCE = 3

RELATES_TO_VOCAB = {
    "addresses",      # Method -> Problem | Research Goal
    "resolves",       # Decision -> Issue | Open Question
    "produces",       # Decision -> Action Item
    "evaluates",      # Experiment | Metrics Evaluation -> Result
    "uses",           # Method | Experiment -> Dataset
    "motivates",      # Background | Research Gap | Problem -> Method | Research Goal
    "identifies",     # Background | Existing Research -> Research Gap
    "extends",        # Method -> Existing Research | Theoretical Basis
    "compares_to",    # Result -> Existing Research | Result
    "contradicts",    # Result -> Result, or Feedback -> Idea | Decision
    "responds_to",    # Feedback -> Idea | Decision | Progress Update | Method | Result | Limitation
    "follows_up_on",  # Progress Update -> Progress Update | Action Item
}

# Source-Type-name(s) -> target-Type-name(s) allowed for each relates_to value
# (v8 ontology). Names are lowercase; matched against canonicalized Type ids.
# A relates_to edge whose endpoints don't fall in these sets is dropped by
# sanitize_graph even if the `relation` string itself is in RELATES_TO_VOCAB.
RELATES_TO_TYPE_PAIRS = {
    "addresses":     ({"method"}, {"problem", "research goal"}),
    "resolves":      ({"decision"}, {"issue", "open question"}),
    "produces":      ({"decision"}, {"action item"}),
    "evaluates":     ({"experiment", "metrics evaluation"}, {"result"}),
    "uses":          ({"method", "experiment"}, {"dataset"}),
    "motivates":     ({"background", "research gap", "problem"}, {"method", "research goal"}),
    "identifies":    ({"background", "existing research"}, {"research gap"}),
    "extends":       ({"method"}, {"existing research", "theoretical basis"}),
    "compares_to":   ({"result"}, {"existing research", "result"}),
    "contradicts":   ({"result", "feedback"}, {"result", "idea", "decision"}),
    "responds_to":   ({"feedback"}, {"idea", "decision", "progress update", "method", "result", "limitation"}),
    "follows_up_on": ({"progress update"}, {"progress update", "action item"}),
}

ACTION_ITEM_TYPE = "Action Item"

# v8.5: the Type for talk that runs the meeting rather than carries its content
# (audio checks, screen sharing, scheduling, opening/closing). Before it existed
# the model had to force such a Topic into Issue/Idea/Feedback — an audio problem
# became an Issue sitting next to the research ones. A Topic typed ONLY with this
# is barred from has_source in sanitize_graph; see `topic_is_procedural`.
MEETING_PROCEDURE_TYPE = "Meeting Procedure"

ALLOWED_TOPIC_STATUS = {"open", "in_progress", "done", "blocked"}
ALLOWED_STANCE = {"raised", "proposed", "decided", "reported", "gave_feedback"}
ALLOWED_CONTRADICTION_LEVEL = {"direct", "partial", "apparent"}

# A Contradiction needs >=2 distinct Description participants to be meaningful.
# Not enforced here — sanitize_graph only sees one chunk at a time — but
# downstream by KnowledgeGraph._cleanup_singleton_contradictions once the whole
# document is stored (see that method's docstring).
MIN_CONTRADICTION_PARTICIPANTS = 2

# --- supersedes (v8.2): edge-only conflict resolution -----------------------
# A corrected / newer Result Description that REPLACES an earlier one — "update
# information", not an unresolved standoff. Unlike Contradiction it creates NO
# node: just a directed edge (newer -> older) between two Result Descriptions.
#
# Produced by the SEPARATE, on-demand conflict-detection pass over the whole KB
# (see docs/conflict_ontology.md), NOT by per-chunk construction — the two
# endpoints usually live in different chunks/documents, which the per-chunk
# sanitizer cannot see. So it is intentionally NOT added to `_FIXED_RELATION_DIRS`
# above (that map is the construction ontology). Its direction, Result-only
# endpoints, and anti-cycle rule (A supersedes B AND B supersedes A must never
# co-exist — that would be a Contradiction, not a supersession) are enforced by
# that detection pass / a downstream whole-graph query, the same way the
# >=2-participant Contradiction rule is enforced downstream rather than in
# `sanitize_graph`.
# DEAD (currently unused — grep confirms no importer). This is the pre-write
# (lowercase) casing only, consistent with `_FIXED_RELATION_DIRS` above — it is
# NOT the casing actually stored in Neo4j for a `supersedes` edge, which is
# written by raw Cypher in src/conflict/classifier.py and must be UPPERCASE
# (see src/conflict/config.py's SUPERSEDES_TYPE for why). Left in place rather
# than removed (unrelated churn), but do not import this into post-write Cypher
# — that would silently reintroduce the casing bug it once caused.
SUPERSEDES_RELATION = "supersedes"
SUPERSEDES_ENDPOINT_TYPE = "Result"   # both Description endpoints must be typeName == "Result"

_PAPER_TYPES = {
    "background", "problem", "research goal", "theoretical basis", "dataset",
    "conclusion", "future work", "existing research", "research gap",
    "method", "experiment", "result", "metrics evaluation", "limitation",
}
_MEETING_TYPES = {
    # v8: discussion-flow types — what a segment of the meeting *is*.
    "issue", "idea", "decision", "action item", "open question",
    "progress update", "feedback",
    # v8.5: the mechanics of running the meeting, not its content. Kept in this
    # set (rather than left to fall through to "unknown") so its Type node is
    # tagged domain="meeting" like the rest — the has_source ban that makes it
    # useful is enforced separately, in sanitize_graph.
    "meeting procedure",
    # v8.4: argumentation / reasoning types — what a segment *asserts* and the
    # reasoning around it. Deliberately still Types (a Topic's function), not new
    # node labels: they classify, they are not content. Note none of these appear
    # in RELATES_TO_TYPE_PAIRS yet, so a Topic typed with one of them can only be
    # linked structurally (has_subtopic) until that table is extended.
    "claim", "observation", "rationale", "assumption", "heuristic",
    "prediction", "option", "evidence", "constraint", "risk",
    "trade off", "exception", "uncertainty", "condition", "rule",
    "action", "outcome", "disagreement",
}


def _type_domain(type_name: str) -> str:
    """
    Deterministic domain tag for a Type node: "paper", "meeting", or "unknown".
    v8's Paper/Meeting vocabularies are disjoint by design (unlike v7, which had
    a "shared" bucket) — a Type name the model invented that matches neither
    list is tagged "unknown" rather than assumed shared.
    """
    key = type_name.strip().lower()
    if key in _PAPER_TYPES:
        return "paper"
    if key in _MEETING_TYPES:
        return "meeting"
    return "unknown"


def classify_expected_domain(source_kind: Optional[str]) -> Optional[str]:
    """
    Deterministic doc-level domain from `source_kind` — NOT from `doc_type`.

    `source_kind` is a controlled enum set by producer code ("pdf" from the
    academic-pdf-chunker, "transcript" from SOURCE_KIND in
    src/ingestion/meeting_chunks.py), unlike `doc_type`, which is a free-text
    folder name ("gold_b1" does not literally contain "paper"). The folder name
    is good enough as a prompt hint; it is not good enough to delete data with,
    which is what the caller does with this value.

    Returns None when the source_kind is neither — callers must not enforce a
    domain in that case.
    """
    if source_kind == "pdf":
        return "paper"
    if source_kind == "transcript":
        return "meeting"
    return None


_MIN_SANE_YEAR = 1900
_MAX_SANE_YEAR = 2100

# Taiwanese ROC/Minguo calendar: Gregorian year = ROC year + 1911. ROC years
# are written digit-by-digit in Chinese numerals ("一一零" = "1","1","0" = 110),
# not as a hundred-based reading — this corpus's theses use this convention
# (e.g. "中華民國一一零年八月"), so digit-by-digit mapping is what's needed, not
# a general Chinese-numeral parser.
_ROC_MARKER_RE = re.compile(r"民國\s*([0-9〇零一二三四五六七八九]+)\s*年")
_CHINESE_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                   "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _parse_source_year(raw: str) -> Optional[int]:
    """
    Best-effort year extraction from an LLM-emitted Source `date` string.
    Returns None (never a guess) if nothing parses or the result fails the
    1900-2100 sanity bound — `date_raw` is preserved by the caller regardless.

    Handles: plain years ("2021"), ISO-ish ("2021-08", "2021-08-15"), English
    month-year ("August 2021"), and ROC/Minguo dates in Arabic ("民國110年8月")
    or Chinese-numeral ("中華民國一一零年八月") form.
    """
    if not raw:
        return None

    roc_match = _ROC_MARKER_RE.search(raw)
    if roc_match:
        digits = roc_match.group(1)
        if digits.isascii() and digits.isdigit():
            roc_year = int(digits)
        else:
            try:
                roc_year = int("".join(str(_CHINESE_DIGITS[ch]) for ch in digits))
            except KeyError:
                roc_year = None
        if roc_year is not None:
            return _bounded_year(roc_year + 1911, raw)
        return None

    year_match = _YEAR_RE.search(raw)
    if year_match:
        return _bounded_year(int(year_match.group(0)), raw)

    return None


def _bounded_year(year: int, raw: str) -> Optional[int]:
    if _MIN_SANE_YEAR <= year <= _MAX_SANE_YEAR:
        return year
    logger.warning(f"Parsed Source year {year} from {raw!r} is outside {_MIN_SANE_YEAR}-{_MAX_SANE_YEAR} — treating as a misparse, dropping.")
    return None


def _capture_source_metadata(raw_properties: Dict[str, Any], source_meta: Dict[str, Any]) -> None:
    """
    First-write-wins accumulation of a Source's `date`/`format` across chunks.
    Mutates `source_meta` in place; never clears a key that's already set, so a
    later chunk with no date (or a conflicting one) can't wipe an earlier
    chunk's captured value. Called once per raw Source node instance seen.
    """
    raw_format = (raw_properties or {}).get("format")
    if raw_format and "format" not in source_meta:
        source_meta["format"] = str(raw_format).strip()

    raw_date = (raw_properties or {}).get("date")
    if not raw_date:
        return
    raw_date = str(raw_date).strip()
    if not raw_date:
        return

    if "date_raw" not in source_meta:
        source_meta["date_raw"] = raw_date
        year = _parse_source_year(raw_date)
        if year is not None:
            # _Node.properties is Dict[str, str] (pydantic-enforced) — every
            # property is a string, same convention as e.g. `due_date` elsewhere.
            source_meta["year"] = str(year)
    elif source_meta["date_raw"] != raw_date:
        logger.warning(
            f"Source date conflict: keeping first-seen {source_meta['date_raw']!r}, "
            f"ignoring later chunk's {raw_date!r}."
        )


def _expected_direction(rel_type: str):
    """Return the (source_label, target_label) a relationship type must have, or None to drop it."""
    rt = (rel_type or "").lower()
    if "::" in rt or " " in rt:            # malformed name (e.g. has_description::x::y)
        return None
    if rt in _FIXED_RELATION_DIRS:
        return _FIXED_RELATION_DIRS[rt]
    if rt.endswith("_description") or rt.endswith("_desc"):
        return ("Type", "Description")     # has_[type]_description : Type -> Description
    if rt.startswith("has_"):
        return ("Topic", "Type")           # has_[type] : Topic -> Type
    return None                            # unknown relationship -> drop


def _resolve_abbreviation_aliases(
    nodes: List[_Node], relationships: List[_Relationship]
) -> Tuple[List[_Node], List[_Relationship]]:
    """
    Merge a bare-abbreviation node into its full-name counterpart when the
    full-name node carries a matching `abbreviation` property.

    Example: a node id "Rmfs" is merged into "Robotic Mobile Fulfillment System
    Rmfs" (properties: {"abbreviation": "RMFS"}). Only catches cases where the
    `abbreviation` property was set; wording variants without an abbreviation
    marker (e.g. "Digital Twin" vs "Digital Twin System") need fuzzy matching,
    which is intentionally out of scope here (false-positive risk).
    """
    abbrev_to_full: Dict[str, str] = {}
    for n in nodes:
        abbr = (n.properties or {}).get("abbreviation")
        if abbr:
            abbrev_to_full[abbr.strip().lower()] = n.id

    if not abbrev_to_full:
        return nodes, relationships

    id_redirect: Dict[str, str] = {}
    for n in nodes:
        full_id = abbrev_to_full.get(n.id.strip().lower())
        if full_id and full_id != n.id:
            id_redirect[n.id] = full_id

    if not id_redirect:
        return nodes, relationships

    kept_nodes = [n for n in nodes if n.id not in id_redirect]

    def remap(node_id: str) -> str:
        return id_redirect.get(node_id, node_id)

    seen = set()
    remapped_rels: List[_Relationship] = []
    for r in relationships:
        src, tgt = remap(r.source), remap(r.target)
        if src == tgt:
            continue  # merging created a self-loop
        key = (src, tgt, r.type.lower())
        if key in seen:
            continue
        seen.add(key)
        remapped_rels.append(
            _Relationship(source=src, target=tgt, type=r.type, properties=r.properties or {})
        )

    return kept_nodes, remapped_rels


def _dedupe_similar_topics(
    nodes: List[_Node],
    relationships: List[_Relationship],
    topic_registry: Optional[Dict[str, str]] = None,
    similarity_threshold: float = 0.92,
) -> Tuple[List[_Node], List[_Relationship]]:
    """
    Merge a Topic node into an already-registered Topic when their canonical
    ids are near-identical strings (e.g. a plural/typo variant the model
    produced on a later chunk, like "Named Entity Recognitions" vs "Named
    Entity Recognition").

    This is a narrow safety net for near-identical STRINGS, not a semantic
    synonym resolver. "NER" vs "Named Entity Recognition" is already handled
    by `_resolve_abbreviation_aliases` above (it needs the explicit
    `abbreviation` property, not string similarity). Genuine synonyms with no
    shared abbreviation and low string overlap — "Digital Twin" vs "Digital
    Twin System" — are intentionally NOT merged here: past ~0.9 similarity,
    false-positive merges (silently collapsing two distinct Topics) become
    more costly than the duplicate they'd prevent. That class of merge needs
    embedding-based matching with human review, which is out of scope for a
    deterministic sanitizer.

    `topic_registry`: a mutable dict the caller creates ONCE per document and
    passes into every chunk's call (same pattern as `has_source_state`), so a
    Topic introduced in chunk 1 is recognized when it reappears with slightly
    different spelling in chunk 5. If None, dedup only happens within this
    single call (per-chunk).
    """
    registry = topic_registry if topic_registry is not None else {}

    id_redirect: Dict[str, str] = {}
    for n in nodes:
        if n.type.capitalize() != "Topic":
            continue
        cid = _canonical_id(n.id)
        match = None
        for existing in registry:
            if difflib.SequenceMatcher(None, cid.lower(), existing.lower()).ratio() >= similarity_threshold:
                match = existing
                break
        if match:
            if cid != match:
                id_redirect[n.id] = match
        else:
            registry[cid] = cid

    if not id_redirect:
        return nodes, relationships

    kept_nodes = [n for n in nodes if n.id not in id_redirect]

    def remap(node_id: str) -> str:
        return id_redirect.get(node_id, node_id)

    seen = set()
    remapped_rels: List[_Relationship] = []
    for r in relationships:
        src, tgt = remap(r.source), remap(r.target)
        if src == tgt:
            continue  # merging created a self-loop
        key = (src, tgt, r.type.lower())
        if key in seen:
            continue
        seen.add(key)
        remapped_rels.append(
            _Relationship(source=src, target=tgt, type=r.type, properties=r.properties or {})
        )

    return kept_nodes, remapped_rels


def sanitize_graph(
    graph: _Graph,
    source_name: str,
    has_source_state: Optional[Dict[str, int]] = None,
    topic_registry: Optional[Dict[str, str]] = None,
    source_meta_state: Optional[Dict[str, dict]] = None,
    expected_domain: Optional[str] = None,
) -> Optional[_Graph]:
    """
    Deterministically enforce the ontology on a model-extracted `_Graph`.

    Repairs the failure modes small models exhibit despite the prompt:
    reversed/hybrid relationship directions, self-loops, malformed relationship
    names, empty Descriptions, duplicate/variant Source nodes, bare-abbreviation
    aliases, near-duplicate Topic strings, and placeholder nodes whose id equals
    their label. All ids are canonicalized so variants merge.

    Also enforces the v8 ontology additions on top of the base structure:
    - `has_[type]` / `has_[type]_description` edges are canonicalized to the
      fixed names `has_type` / `has_description` regardless of what the model
      actually emitted (e.g. a leaked "has_method" is renamed, not just
      direction-validated) — this is what makes "10 fixed relationship names"
      true even when the model doesn't comply with the prompt.
    - `relates_to` (Topic -> Topic): kept only when its `relation` property is
      one of `RELATES_TO_VOCAB` AND the source/target Topics' has_type Types
      match the allowed pair for that relation in `RELATES_TO_TYPE_PAIRS`
      (e.g. "addresses" requires a Method -> Problem/Research Goal pair) — an
      untyped, invented, or mismatched-pair relation is dropped rather than
      trusted.
    - `assigned_to` (Topic -> Agent): kept only when the source Topic has a
      validated has_type edge to the Type "Action Item" — this stops the
      model from attaching an assignee to a Topic that isn't actually an
      action item.
    - `has_source` (Topic -> Source, v8.5): dropped outright when the source
      Topic's ONLY Type is "Meeting Procedure", before the MAX_HAS_SOURCE cap
      is consulted. Keeps a transcript's opening audio-check from claiming the
      document's "core topic" slots ahead of its actual content — see
      `topic_is_procedural` for the full reasoning.
    - `status` (on Topic nodes) and `stance` (on spoke_about/writes_about
      edges) are kept only when the value is one of the allowed options;
      otherwise the property is dropped, not the whole node/edge.
    - `has_contradiction` (Description -> Contradiction, v8.1): direction and
      an empty `summary` on the Contradiction node are enforced like
      Description's own rules. `level` is kept only when it's one of
      ALLOWED_CONTRADICTION_LEVEL, dropped (not the edge) otherwise — same
      leniency as `stance`. The >=2-participants floor is NOT enforced here
      (see MIN_CONTRADICTION_PARTICIPANTS); it runs downstream in
      `KnowledgeGraph._cleanup_singleton_contradictions`.
    - Description ids are scoped per source document (v8.3): a Description's id
      is its content-only canonical id plus `|` + `canon_source`, and it gets a
      `source_id` property set to `canon_source`. Without this, two different
      documents describing the same Topic+Type would MERGE into a single
      Description node on write (storage MERGEs by id), silently conflating two
      documents' distinct claims and making cross-document conflict detection
      (docs/conflict_ontology.md) impossible — same Topic+Type would always
      resolve to exactly one node. See `description_remap` below.

    Every Type node also gets a deterministic `domain` property ("paper",
    "meeting", or "unknown") via `_type_domain`, so the same Type label is not
    a flat namespace mixing paper and meeting vocabularies with no way to tell
    which document kind it came from. v8's Paper/Meeting Type lists are
    disjoint by design, so "unknown" only fires for a Type the model invented.

    `has_source_state`: a mutable dict the caller creates ONCE per document and
    passes into every chunk's call, so the "max 3 has_source edges" cap spans the
    whole document instead of resetting per chunk. If None, the cap applies within
    this single call only (per-chunk — the pre-fix behavior).

    `topic_registry`: a mutable dict the caller creates ONCE per document and
    passes into every chunk's call, so near-duplicate Topic strings merge across
    chunks instead of only within one. If None, dedup applies within this single
    call only. See `_dedupe_similar_topics`.

    `source_meta_state`: a mutable dict the caller creates ONCE per document and
    passes into every chunk's call (same pattern as the two above), so a `date`/
    `format` the LLM emits on a Source node in one chunk is still present on
    every later chunk's write of the same canonical Source — MERGE keeps
    re-applying it, so it's never lost even though the raw Source node is
    re-emitted (and its properties otherwise discarded for alias-collapsing
    purposes) on every chunk. First-write-wins on a date conflict between
    chunks; see `_capture_source_metadata`. If None, capture is scoped to this
    single call only.

    `expected_domain`: "paper", "meeting", or None — the domain the DOCUMENT is,
    derived from its `source_kind` by `classify_expected_domain`. Unlike the
    three above it is stateless. Only one direction is enforced:
    `expected_domain == "paper"` drops every meeting-domain Type node (and the
    Description hanging off it), because a PDF paper can never legitimately
    produce one. The reverse is deliberately NOT enforced — a meeting reviewing
    paper progress genuinely does produce paper-domain Types (Feedback on a
    Method), which the prompt explicitly allows and the real meeting corpus
    contains. None disables the check entirely, which is what an unrecognised
    source_kind must get.
    """
    if graph is None:
        return None

    canon_source = _canonical_id(source_name)
    source_meta = source_meta_state.setdefault(canon_source, {}) if source_meta_state is not None else {}

    # 0. Drop empty/placeholder Descriptions and Contradictions (and any edge
    #    touching them) up front — a Contradiction with an empty `summary` is
    #    as useless as an empty Description, regardless of its edge count.
    drop_ids = {
        n.id for n in graph.nodes
        if n.type.capitalize() == "Description" and not (n.properties or {}).get("text", "").strip()
    } | {
        n.id for n in graph.nodes
        if n.type.capitalize() == "Contradiction" and not (n.properties or {}).get("summary", "").strip()
    }
    nodes = [n for n in graph.nodes if n.id not in drop_ids]
    relationships = [
        r for r in graph.relationships
        if r.source not in drop_ids and r.target not in drop_ids
    ]

    # 1. Merge bare-abbreviation nodes into their full-name counterparts.
    nodes, relationships = _resolve_abbreviation_aliases(nodes, relationships)

    # 1b. Merge near-duplicate Topic strings (see _dedupe_similar_topics for
    #     why this stops short of semantic synonym resolution).
    nodes, relationships = _dedupe_similar_topics(nodes, relationships, topic_registry)

    # 1c. Domain enforcement (only when the caller knows the document's domain).
    #     A paper source emitting a meeting-domain Type is always a leak, so the
    #     Type node goes — and with it the Description it owns, which would
    #     otherwise survive as an orphan with no incoming has_description and
    #     still be picked up by anything that scans Description nodes directly
    #     (e.g. the conflict pass). Dropping the Type alone is not enough.
    #
    #     A Description shared with a Type that IS kept is spared: Description
    #     ids are Topic+Type specific so this should not happen, but losing a
    #     surviving Type's only Description is worse than keeping one extra node.
    domain_drop_ids: Set[str] = set()
    if expected_domain == "paper":
        kept_type_ids, dropped_type_ids = set(), set()
        for n in nodes:
            if n.type.capitalize() != "Type":
                continue
            cid = _canonical_id(n.id)
            (dropped_type_ids if _type_domain(cid) == "meeting" else kept_type_ids).add(cid)

        if dropped_type_ids:
            desc_of_dropped, desc_of_kept = set(), set()
            for r in relationships:
                rt = (r.type or "").lower()
                if not (rt.endswith("_description") or rt.endswith("_desc")):
                    continue
                src_cid, tgt_cid = _canonical_id(r.source), _canonical_id(r.target)
                if src_cid in dropped_type_ids:
                    desc_of_dropped.add(tgt_cid)
                elif src_cid in kept_type_ids:
                    desc_of_kept.add(tgt_cid)
            domain_drop_ids = dropped_type_ids | (desc_of_dropped - desc_of_kept)
            logger.info(
                f"{source_name}: dropping {len(dropped_type_ids)} meeting-domain Type(s) "
                f"from a paper source: {', '.join(sorted(dropped_type_ids))}"
            )

    # 2. Keep only ontology-labelled nodes; drop placeholders (id == label).
    #    Collapse every Source-labelled node into one canonical Source.
    #
    #    Description ids get an extra doc-scoping step here (v8.3): a Description's
    #    content-only id (`cid`, e.g. "Description Multimodal Retrieval Result") is
    #    identical for two different documents that happen to discuss the same
    #    Topic+Type. Since storage MERGEs by (label, id), that collision silently
    #    folds two documents' distinct claims into one node — which also means the
    #    whole-KB conflict pass in docs/conflict_ontology.md could never find a
    #    cross-document pair for a shared Topic+Type: there would only ever be one
    #    node to find. `description_remap` (cid -> doc-scoped id) fixes this by
    #    suffixing every Description id with `canon_source`, joined with "|" rather
    #    than "::" — "::" contains ":", which `_canonical_id` strips, and this
    #    function is applied to node ids a SECOND time downstream (`map_to_lc_node`,
    #    and `bench/graph_metrics.py`'s own re-canonicalization), which would
    #    silently collapse a "::" suffix back into a space and reopen the same
    #    collision. "|" survives both passes unchanged.
    node_label: Dict[str, str] = {}
    out_nodes: List[_Node] = []
    source_aliases = set()
    description_remap: Dict[str, str] = {}
    for n in nodes:
        label = n.type.capitalize()
        if label not in ALLOWED_LABELS:
            continue
        cid = _canonical_id(n.id)
        if not cid or cid == label:
            continue
        if label == "Type" and cid.lower().startswith("has "):
            continue  # relationship name leaked in as a Type node
        if cid in domain_drop_ids:
            # Wrong-domain Type (or its now-orphaned Description) — see 1c. It
            # never enters `node_label`, so every has_type / has_description
            # edge touching it fails the endpoint check in step 3 and is dropped
            # with it; no separate edge filtering is needed.
            continue
        if label == "Source":
            source_aliases.add(cid)
            _capture_source_metadata(n.properties or {}, source_meta)
            continue
        if label == "Description":
            doc_scoped_id = f"{cid}|{canon_source}"
            description_remap[cid] = doc_scoped_id
            cid = doc_scoped_id
        if cid not in node_label:
            node_label[cid] = label
            props = {**(n.properties or {}), "name": cid}
            if label == "Type":
                props["domain"] = _type_domain(cid)
            if label == "Description":
                props["source_id"] = canon_source
            if label == "Topic" and "status" in props:
                if str(props["status"]).strip().lower() not in ALLOWED_TOPIC_STATUS:
                    del props["status"]  # invented value — drop rather than trust it
            out_nodes.append(_Node(id=cid, type=label, properties=props))

    node_label[canon_source] = "Source"
    out_nodes.append(_Node(id=canon_source, type="Source", properties={"name": canon_source, **source_meta}))

    def resolve(node_id: str) -> str:
        cid = _canonical_id(node_id)
        if cid in source_aliases:
            return canon_source
        return description_remap.get(cid, cid)

    # 3. Validate relationships against the direction whitelist. The has_source
    #    cap is tracked in `has_source_state` (keyed by canonical Source id) so it
    #    holds across every chunk of the document, not just this single call.
    counter = has_source_state if has_source_state is not None else {}
    has_source_count = counter.get(canon_source, 0)

    # Pre-pass: which Type(s) is each Topic validly linked to via has_[type]?
    # Needed both to gate assigned_to (Action Item only) and to enforce the
    # relates_to type-pair table below. Computed before the main loop since a
    # chunk may list relates_to/assigned_to before its has_[type] edge.
    topic_type_names: Dict[str, set] = {}
    for r in relationships:
        rt_low = (r.type or "").lower()
        if not rt_low.startswith("has_") or rt_low.endswith("_description") or rt_low.endswith("_desc"):
            continue
        src, tgt = resolve(r.source), resolve(r.target)
        if node_label.get(src) == "Topic" and node_label.get(tgt) == "Type":
            topic_type_names.setdefault(src, set()).add(tgt.strip().lower())

    topic_is_action_item = {
        topic for topic, types in topic_type_names.items()
        if ACTION_ITEM_TYPE.lower() in types
    }

    # Topics that are ONLY procedural — the meeting's mechanics, not its content.
    # `== {…}` and not `in`: a Topic typed both Meeting Procedure and Decision is
    # a real decision that happens to be about process, and keeps its claim on a
    # has_source slot.
    #
    # WHY they are barred from has_source: the MAX_HAS_SOURCE cap below is
    # first-come-first-served in chunk order (has_source_state spans the whole
    # document), and a transcript's chunk 1 is greetings and "can you hear me?".
    # Left alone it fills all three "document core" slots before the first chunk
    # of real content is even reached, and every later Topic's has_source is
    # dropped. That ordering happens to work for papers, whose chunk 1 is the
    # abstract; it inverts on meetings.
    topic_is_procedural = {
        topic for topic, types in topic_type_names.items()
        if types == {MEETING_PROCEDURE_TYPE.lower()}
    }

    out_rels: List[_Relationship] = []
    seen = set()
    for r in relationships:
        src = resolve(r.source)
        tgt = resolve(r.target)
        if src == tgt:                      # self-loop
            continue
        expected = _expected_direction(r.type)
        if expected is None:
            continue
        if node_label.get(src) != expected[0] or node_label.get(tgt) != expected[1]:
            continue
        rt = r.type.lower()

        # Canonicalize has_[type] / has_[type]_description variants to the
        # fixed v8 edge names regardless of what the model actually emitted —
        # this is what makes "10 fixed relationship names" true even when the
        # model reverts to dynamic naming like "has_method".
        if expected == ("Topic", "Type"):
            rt = "has_type"
        elif expected == ("Type", "Description"):
            rt = "has_description"

        props = dict(r.properties or {})
        if rt == "relates_to":
            relation = (props.get("relation") or "").strip().lower().replace(" ", "_")
            if relation not in RELATES_TO_VOCAB:
                continue  # untyped/invented relation — drop rather than trust verbatim
            pair = RELATES_TO_TYPE_PAIRS.get(relation)
            if pair is not None:
                src_types = topic_type_names.get(src, set())
                tgt_types = topic_type_names.get(tgt, set())
                if not (src_types & pair[0]) or not (tgt_types & pair[1]):
                    continue  # endpoint Types don't match this relation's allowed pair
            props["relation"] = relation
        if rt == "assigned_to" and src not in topic_is_action_item:
            continue  # assignee attached to a Topic that isn't an Action Item
        if rt in ("spoke_about", "writes_about") and "stance" in props:
            if str(props["stance"]).strip().lower() not in ALLOWED_STANCE:
                del props["stance"]  # invented value — drop rather than trust it
        if rt == "has_contradiction":
            level = str(props.get("level", "")).strip().lower()
            if level in ALLOWED_CONTRADICTION_LEVEL:
                props["level"] = level
            else:
                # Drop the bad value but KEEP the edge — unlike relates_to, which
                # drops the whole edge for a bad `relation`. Dropping it here could
                # push a genuine Contradiction below the 2-participant floor over a
                # mere formatting slip; that floor is enforced downstream instead.
                props.pop("level", None)

        key = (src, tgt, rt)
        if key in seen:                     # dedup before counting so duplicates don't eat the cap
            continue
        if rt == "has_source":
            # Checked before the cap so a procedural Topic does not merely lose
            # its own edge — it never consumes a slot a content Topic could use.
            if src in topic_is_procedural:
                continue
            if has_source_count >= MAX_HAS_SOURCE:
                continue
            has_source_count += 1
        seen.add(key)
        out_rels.append(_Relationship(source=src, target=tgt, type=rt, properties=props))

    counter[canon_source] = has_source_count  # persist for the caller's next chunk

    return _Graph(nodes=out_nodes, relationships=out_rels)


def map_to_lc_node(node: _Node) -> Node:
    """Maps the `_Graph` `_Node` to the `langchain_neo4j.graphs.graph_document.Node`"""
    properties = node.properties if node.properties else {}
    canonical = _canonical_id(node.id)
    # Add name property for better Cypher statement generation
    properties["name"] = canonical
    return Node(
        id=canonical,
        type=node.type.capitalize(),
        properties=properties
    )


def map_to_lc_relationship(rel: _Relationship, nodes: List[_Node]) -> Relationship:
    """Maps the `_Graph` `_Relationship`  to the `langchain_neo4j.graphs.graph_document.Relationship`"""
    
    source_node = [node for node in nodes if node.id == rel.source][0]
    target_node = [node for node in nodes if node.id == rel.target][0]

    source = map_to_lc_node(source_node)
    target = map_to_lc_node(target_node)

    properties = rel.properties if rel.properties else {}

    return Relationship(
        source=source, 
        target=target, 
        type=rel.type, 
        properties=properties
    )


def map_to_lc_graph(graph: _Graph, source_content: str) -> GraphDocument:
    """
    Maps the `_Graph` class to the 
    `langchain_neo4j.graphs.graph_document.GraphDocuemnt` class
    """
    valid_nodes = [node for node in graph.nodes if node.type.capitalize() in ALLOWED_LABELS]
    valid_ids = {node.id for node in valid_nodes}

    nodes = [map_to_lc_node(node) for node in valid_nodes]

    relationships = [
        map_to_lc_relationship(rel, valid_nodes)
        for rel in graph.relationships
        if rel.source in valid_ids and rel.target in valid_ids
    ]

    graph_doc = GraphDocument(
        nodes=nodes, 
        relationships=relationships,
        source=Document(page_content=source_content)
    )

    return graph_doc