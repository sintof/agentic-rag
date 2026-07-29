from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    tavily_api_key: str = ""

    qdrant_url: str = ""
    qdrant_collection: str = "agentic_rag_docs"

    file_storage_path: str = "./uploads"
    max_generation_retries: int = 2

    max_upload_bytes: int = 15 * 1024 * 1024  # 15MB
    max_images_per_document: int = 20
    max_extracted_chars: int = 300_000

    proxy_base_url: str = "https://saidazam-litellm-proxy.hf.space/v1"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
