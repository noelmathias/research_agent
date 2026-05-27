from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout_seconds: float = 300.0

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # App
    app_name: str = "Research Agent"
    debug: bool = True

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # ChromaDB
    chroma_persist_dir: str = "data/vectorstore"
    chroma_collection_name: str = "research_docs"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Uploads
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 50

    # Evaluation / Reflection
    confidence_threshold: float = 0.5
    max_retries: int = 2

    # Reports
    reports_dir: str = "data/reports"

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

