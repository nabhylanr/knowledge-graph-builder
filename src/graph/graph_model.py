import re
import networkx as nx

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

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


class Ontology(BaseModel):
    """     
    Used to describe arbitrary, project-specific allowed labels and relationships.

    Labels should map to `Node.type` and relationships to `Relationship.type` from 
    `langchain_neo4j.graphs.graph_document`. It is allowed to provide a functional 
    description of what labels and relationships represent in the domain.  
    
    """
    allowed_labels: Optional[List[str]]=None
    labels_descriptions: Optional[Dict[str, str]]=None
    allowed_relations: Optional[List[str]]=None


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


# The six node labels allowed by the extraction ontology. Anything else
# (e.g. a relationship name emitted as a node) is dropped during mapping.
ALLOWED_LABELS = {"Agent", "Role", "Topic", "Type", "Source", "Description"}


def _canonical_id(s: str) -> str:
    """
    Canonical form of a node id so that casing / underscore / spacing variants
    collapse into a single node on MERGE (e.g. "Research_Problem" and
    "Research Problem" both become "Research Problem").
    """
    return re.sub(r"\s+", " ", _normalize(s)).strip().title()


# Allowed relationship directions as (source_label, target_label) per relationship
# type. Small models frequently reverse these or invent hybrids; the sanitizer
# enforces them deterministically instead of trusting the prompt.
_FIXED_RELATION_DIRS = {
    "role_in_meeting": ("Agent", "Role"),
    "role_in_paper": ("Agent", "Role"),
    "spoke_about": ("Agent", "Topic"),
    "writes_about": ("Agent", "Topic"),
    "has_source": ("Topic", "Source"),
    "has_subtopic": ("Topic", "Topic"),
}

MAX_HAS_SOURCE = 3


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


def sanitize_graph(
    graph: _Graph,
    source_name: str,
    has_source_state: Optional[Dict[str, int]] = None,
) -> Optional[_Graph]:
    """
    Deterministically enforce the ontology on a model-extracted `_Graph`.

    Repairs the failure modes small models exhibit despite the prompt:
    reversed/hybrid relationship directions, self-loops, malformed relationship
    names, empty Descriptions, duplicate/variant Source nodes, bare-abbreviation
    aliases, and placeholder nodes whose id equals their label. All ids are
    canonicalized so variants merge.

    `has_source_state`: a mutable dict the caller creates ONCE per document and
    passes into every chunk's call, so the "max 3 has_source edges" cap spans the
    whole document instead of resetting per chunk. If None, the cap applies within
    this single call only (per-chunk — the pre-fix behavior).
    """
    if graph is None:
        return None

    canon_source = _canonical_id(source_name)

    # 0. Drop empty/placeholder Descriptions (and any edge touching them) up front.
    drop_ids = {
        n.id for n in graph.nodes
        if n.type.capitalize() == "Description" and not (n.properties or {}).get("text", "").strip()
    }
    nodes = [n for n in graph.nodes if n.id not in drop_ids]
    relationships = [
        r for r in graph.relationships
        if r.source not in drop_ids and r.target not in drop_ids
    ]

    # 1. Merge bare-abbreviation nodes into their full-name counterparts.
    nodes, relationships = _resolve_abbreviation_aliases(nodes, relationships)

    # 2. Keep only ontology-labelled nodes; drop placeholders (id == label).
    #    Collapse every Source-labelled node into one canonical Source.
    node_label: Dict[str, str] = {}
    out_nodes: List[_Node] = []
    source_aliases = set()
    for n in nodes:
        label = n.type.capitalize()
        if label not in ALLOWED_LABELS:
            continue
        cid = _canonical_id(n.id)
        if not cid or cid == label:
            continue
        if label == "Type" and cid.lower().startswith("has "):
            continue  # relationship name leaked in as a Type node
        if label == "Source":
            source_aliases.add(cid)
            continue
        if cid not in node_label:
            node_label[cid] = label
            props = {**(n.properties or {}), "name": cid}
            out_nodes.append(_Node(id=cid, type=label, properties=props))

    node_label[canon_source] = "Source"
    out_nodes.append(_Node(id=canon_source, type="Source", properties={"name": canon_source}))

    def resolve(node_id: str) -> str:
        cid = _canonical_id(node_id)
        return canon_source if cid in source_aliases else cid

    # 3. Validate relationships against the direction whitelist. The has_source
    #    cap is tracked in `has_source_state` (keyed by canonical Source id) so it
    #    holds across every chunk of the document, not just this single call.
    counter = has_source_state if has_source_state is not None else {}
    has_source_count = counter.get(canon_source, 0)

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
        key = (src, tgt, rt)
        if key in seen:                     # dedup before counting so duplicates don't eat the cap
            continue
        if rt == "has_source":
            if has_source_count >= MAX_HAS_SOURCE:
                continue
            has_source_count += 1
        seen.add(key)
        out_rels.append(_Relationship(source=src, target=tgt, type=rt, properties=r.properties or {}))

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
    # Drop nodes whose type is outside the ontology (e.g. a relationship name
    # emitted as a node by a weaker model), then keep only relationships whose
    # endpoints both survive — this also guards against the [0] lookup below
    # raising on a dangling relationship.
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