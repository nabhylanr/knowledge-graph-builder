from enum import Enum
from pydantic import BaseModel
from typing import Optional

from src.graph.graph_model import Ontology


class ModelType(str, Enum):
    """
    Type of LLM / embedding providers available in the toolkit.
    """
    AZURE_OPENAI = "azure-openai"
    GOOGLE = "google"
    GROQ = "groq"
    OPENAI = "openai"
    OLLAMA = "ollama"
    TRANSFORMERS = "trf"


class LLMConf(BaseModel):
    """
    Configuration for an LLM.
    -----------
    attributes:
    -----------
    `type`: LLM `ModelType`
    `temperature`: LLM temperature param
    `deployment`: represents the name of the deployment.
    `model`: represents the name of the model
    `api_key`: reference to the OpenAI (or Groq, or Azure OpenAI) API key, if any
    `endpoint`: reference to the endpoint of the model, if any
    """
    model: str
    temperature: float = 0.0
    type: ModelType = "groq"
    deployment: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: Optional[str] = None


class EmbedderConf(BaseModel):
    """
    Embeddings model configuration.
    -----------
    attributes:
    -----------
    `type`: embedder `ModelType`
    `deployment`: represents the name of the deployment.
    `model`: represents the name of the model
    `api_key`: reference to the OpenAI (or Azure OpenAI) API key, if any
    `endpoint`: reference to the endpoint of the model, if any
    """
    type: ModelType = "ollama"
    model: Optional[str] = "mxbai-embed-large"
    deployment: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: Optional[str] = None


class KnowledgeGraphConfig(BaseModel):
    """
    Configuration for the Neo4j backend of the Knowledge Graph.
    It might come with an `Ontology`, a pre-established set of allowed relationships
    and node labels.

    -----------
    attributes:
    -----------
    `password`: `str`
    `db_schema`: `str`
    `host_name`: `str`
    `port`: `int`
    `user`: `str`
    `database`: `str`
    `index_name`: `str`
    `timeout`: `int`
    `ontology`: `Ontology`
    `uri`: `str`
    """
    password: str
    db_schema: Optional[str] = None
    host_name: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    database: Optional[str] = None
    index_name: str = "vector"
    timeout: int = 5000
    ontology: Optional[Ontology] = None
    uri: Optional[str] = None


class Configuration(BaseModel):
    """
    Configuration for the Knowledge Graph building pipeline.

    -----------
    attributes:
    -----------
    `database`: configuration to access the Neo4j Graph Database
    `re_model_conf`: configuration for the LLM in charge of extracting the graph from chunks
    `embedder_conf`: configuration for the Embeddings model that will create vectors out of chunks
    """
    database: KnowledgeGraphConfig
    re_model_conf: Optional[LLMConf] = None
    embedder_conf: Optional[EmbedderConf] = None
