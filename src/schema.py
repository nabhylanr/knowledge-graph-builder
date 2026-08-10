from pydantic import BaseModel
from typing import List, Dict, Optional, Union

from langchain_core.load.serializable import Serializable
from langchain_neo4j.graphs.graph_document import Node, Relationship


class Chunk(BaseModel):
    chunk_id: Union[int, str]
    text: str
    filename: Optional[str] = None
    # False = store this chunk (text + embedding + NEXT chain) but do not spend an
    # LLM call extracting a graph from it. Set by the producer for chunks that
    # carry no extractable content — a transcript's "you" or "Thank you." — which
    # the extractor cannot decline to answer for, since its prompt requires 1-3
    # Topics per chunk. See GraphMiner.mine_graph_from_doc_chunks.
    extraction_eligible: bool = True
    embedding: Optional[List[float]] = None
    chunk_size: int=1000
    chunk_overlap: int=100
    embeddings_model: Optional[str] = None
    nodes: Optional[List[Node]] = None
    relationships: Optional[List[Relationship]] = None


class ProcessedDocument(BaseModel):
    filename: str = ""
    source: str= ""
    document_version: int = 1
    metadata: Optional[dict] = None
    chunks: Optional[List[Chunk]] = None





# class Node(Serializable):
#     id: str
#     type: str
#     properties: Optional[Dict[str, str]] = None


# class Relationship(Serializable):
#     source: str
#     target: str
#     type: str
#     properties: Optional[Dict[str, str]] = None


# class Graph(Serializable):
#     """ 
#     Represents a graph consisting of nodes and relationships.

#     Attributes:
#         nodes (List[Node]): A list of nodes in the graph.
#         relationships (List[Relationship]): A list of relationships in the graph.
#     """
#     nodes: List[Node]
#     relationships: List[Relationship]