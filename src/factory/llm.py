from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq.chat_models import ChatGroq
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI, AzureChatOpenAI
from src.utils.logger import get_logger

from src.config import LLMConf

logger = get_logger(__name__)


def fetch_llm(conf: LLMConf) -> Optional[BaseChatModel]:
    """
    Fetches the LLM model.
    """
    logger.info(f"Fetching LLM model '{conf.model}'..")

    if conf.type == "ollama":
        ollama_kwargs = {"model": conf.model, "temperature": conf.temperature}
        # Point at a REMOTE Ollama server via RE_MODEL_ENDPOINT (e.g. a Tailscale
        # host running the model), so an underpowered local machine offloads
        # inference. Left unset/"none" it defaults to local localhost:11434.
        if conf.endpoint and conf.endpoint.strip().lower() not in ("", "none"):
            ollama_kwargs["base_url"] = conf.endpoint.strip()
        llm = ChatOllama(**ollama_kwargs)
    elif conf.type == "openai":
        llm = ChatOpenAI(
            model=conf.model,
            api_key=conf.api_key,
            deployment=conf.deployment,
            temperature=conf.temperature,
        )
    elif conf.type == "azure-openai":
        llm = AzureChatOpenAI(
            model=conf.model,
            azure_endpoint=conf.endpoint,
            azure_deployment=conf.deployment,
            api_key=conf.api_key,
            temperature=conf.temperature,
            api_version=conf.api_version
        )
    elif conf.type == "groq":
        llm = ChatGroq(
            model=conf.model,
            api_key=conf.api_key,
            temperature=conf.temperature,
            max_retries=3
        )
    elif conf.type == "google":
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=conf.model,
            api_key=conf.api_key,
            temperature=conf.temperature,
        )
    elif conf.type == "trf":
        # Imported lazily (like the Google branch): langchain_huggingface pulls in
        # transformers/torch (~GBs), so only load it when a HF model is actually
        # requested — importing this module for a Groq/OpenAI run stays torch-free.
        from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
        llm = ChatHuggingFace(
            model=conf.model,
            endpoint=conf.endpoint,
            temperature=conf.temperature
        )
    else:
        logger.warning(f"LLM type '{conf.type}' not supported.")
        llm = None

    logger.info(f"Initialized LLM of type: '{conf.type}'")
    return llm