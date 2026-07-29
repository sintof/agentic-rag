"""
All model calls go through the class LiteLLM proxy (OpenAI-compatible), never directly
to Google. Three roles, three model names, per the class's routing rules:

  - gemini-flash-lite : high-volume calls (answer generation, image captioning)
  - gemini-flash       : supervisor/critic calls (document grading, hallucination /
                          relevance grading — the decisions that make this "agentic")
  - gemini-embedding   : embeddings for the vector store
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import settings

MODEL_LITE = "gemini-flash-lite"
# NOTE: this class key is scoped to ['flash-lite', 'gemini-flash-lite', 'gemini-embedding']
# only — "gemini-flash" returns 403 key_model_access_denied. Using flash-lite for the
# critic role too until/unless the key's access is widened.
MODEL_FLASH = "gemini-flash-lite"
MODEL_EMBED = "gemini-embedding"


def _require_key() -> str:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy backend/.env.example to backend/.env and "
            "add the class proxy key."
        )
    return settings.gemini_api_key


def get_llm_lite(temperature: float = 0.2) -> ChatOpenAI:
    """High-volume model: generation, image captioning."""
    return ChatOpenAI(
        base_url=settings.proxy_base_url,
        api_key=_require_key(),
        model=MODEL_LITE,
        temperature=temperature,
    )


def get_llm_flash(temperature: float = 0.0) -> ChatOpenAI:
    """Supervisor/critic model: grading and routing decisions."""
    return ChatOpenAI(
        base_url=settings.proxy_base_url,
        api_key=_require_key(),
        model=MODEL_FLASH,
        temperature=temperature,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=settings.proxy_base_url,
        api_key=_require_key(),
        model=MODEL_EMBED,
    )
