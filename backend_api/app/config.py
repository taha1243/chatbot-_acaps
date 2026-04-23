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
    vllm_model: str = "qwen2.5:7b"  # Ollama model name
    
    # LLM Provider: "ollama", "vllm", "openai", "openrouter"
    llm_provider: str = "openrouter"
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
    # Note: BGE-M3 similarity scores typically range 0.4-0.6, so threshold must be lower
    similarity_threshold: float = 0.40
    top_k_results: int = 5
    max_context_length: int = 8192

    # Generation Settings
    temperature: float = 0.0
    max_tokens: int = 1024

    # Guardrails
    enable_guardrails: bool = False
    # Retrieval confidence below which the output rail blocks as LOW_CONFIDENCE.
    # Distinct from similarity_threshold (which filters candidates during retrieval).
    min_confidence_threshold: float = 0.35
    # Minimum fraction of significant response tokens that must appear in the
    # retrieved context. Below this, the output is flagged as HALLUCINATION.
    groundedness_threshold: float = 0.50
    # Detect emails, phone numbers, CIN, IBAN, credit cards in user input.
    enable_pii_detection: bool = True
    # Semantic off-topic detection using embedding similarity to ACAPS anchors.
    # Catches paraphrases that the regex off-topic patterns miss.
    enable_semantic_scope: bool = True
    # Cosine similarity below which a query is considered off-topic.
    # BGE-M3 on two short related phrases typically scores 0.40-0.60.
    semantic_scope_threshold: float = 0.35

    # Greeting / small-talk routing. The regex pre-filter in engine.py handles
    # the obvious cases; this threshold gates the embedding-based fallback that
    # catches paraphrases and typos the regex misses.
    enable_semantic_greeting: bool = True
    greeting_similarity_threshold: float = 0.70
    
    # CORS
    cors_origins: str = "*"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

