from __future__ import annotations

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from typing import Union

from src.config import EmbedderConf
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_embeddings(conf: EmbedderConf) -> Union[
    HuggingFaceEmbeddings, 
    OllamaEmbeddings, 
    OpenAIEmbeddings, 
    AzureOpenAIEmbeddings, 
    None
    ]:

        if conf.type == "ollama":
            ollama_kwargs = {"model": conf.model}
            # Point at a REMOTE Ollama server via EMBEDDINGS_ENDPOINT (e.g. a
            # Tailscale host), mirroring the LLM factory. Left unset/"none" it
            # defaults to local localhost:11434.
            if conf.endpoint and conf.endpoint.strip().lower() not in ("", "none"):
                ollama_kwargs["base_url"] = conf.endpoint.strip()
            embeddings = OllamaEmbeddings(**ollama_kwargs)
        elif conf.type == "openai":
            embeddings = OpenAIEmbeddings(
                model=conf.model,
                api_key=conf.api_key,
                deployment=conf.deployment,
            )
        elif conf.type == "azure-openai":
            embeddings = AzureOpenAIEmbeddings(
                model=conf.model, 
                azure_endpoint=conf.endpoint,
                azure_deployment=conf.deployment,
                dimensions=1536,
                api_key=conf.api_key,
                api_version=conf.api_version
                
            )
        elif conf.type == "trf":
            # Lazy import: langchain_huggingface pulls transformers/torch (~GBs);
            # only load it when a HF embedder is actually requested.
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model=conf.model,
                endpoint=conf.endpoint,
            )
        else: 
            logger.warning(f"Embedder type '{conf.type}' not supported.")
            embeddings = None

        return embeddings