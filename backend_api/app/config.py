"""
Configuration for Backend API
Uses pydantic-settings for environment variable management.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Atlas-RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # LLM — Ollama via OpenAI-compatible API
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://ollama:11434/v1"
    llm_api_key: str = "ollama"

    # PostgreSQL + pgvector
    database_url: str = "postgresql://atlas:atlas@localhost:5432/atlas_rag"
    vector_table: str = "atlas_knowledge"

    # Embedding Model
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dimension: int = 768

    # RAG Configuration
    similarity_threshold: float = 0.25
    top_k_results: int = 5
    max_context_length: int = 8192

    # Generation Settings
    temperature: float = 0.7
    max_tokens: int = 1024

    # Guardrails
    enable_guardrails: bool = False

    # CORS
    cors_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
