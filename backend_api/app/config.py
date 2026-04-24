"""
Configuration for Backend API
Uses pydantic-settings for environment variable management.
"""
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Atlas-RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # LLM Inference Server (Ollama, vLLM, OpenRouter, or OpenAI-compatible)
    vllm_url: str = "http://localhost:11434/v1"  # Default to Ollama
    vllm_api_key: str = "ollama"  # Ollama doesn't need a real key
    vllm_model: str = "qwen2.5:3b"  # Ollama model name
    
    # LLM Provider: "ollama", "vllm", "openai", "openrouter"
    llm_provider: str = "ollama"
    openrouter_site_url: Optional[str] = None
    openrouter_title: Optional[str] = None
    
    # Fallback: Return context-only response if LLM unavailable
    llm_fallback_enabled: bool = True
    
    # PostgreSQL + pgvector
    database_url: str = "postgresql://atlas:atlas@localhost:5432/atlas_rag"
    vector_table: str = "atlas_knowledge"
    
    # Embedding Model
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    
    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = True
    
    # RAG Configuration
    # BGE-M3 scores depend on chunk size. For section-level chunks (~2k chars),
    # relevant matches typically score 0.25-0.45. Keep threshold low; the intent
    # filter in engine.py already scopes results to the correct section.
    similarity_threshold: float = 0.25
    top_k_results: int = 3
    max_context_length: int = 8192
    
    # Generation Settings
    temperature: float = 0.0
    max_tokens: int = 500
    
    # Guardrails
    enable_guardrails: bool = False
    
    # CORS
    cors_origins: str = "*"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

