"""
Configuration for Data Ingestion Pipeline
"""
import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


@dataclass
class IngestionConfig:
    """Configuration for the data ingestion pipeline."""
    
    # PostgreSQL + pgvector settings
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://atlas:atlas@localhost:5432/atlas_rag"
    )
    vector_table: str = os.getenv("VECTOR_TABLE", "atlas_knowledge")
    
    # Embedding Settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    
    # Processing Settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Paths
    documents_dir: str = os.path.join(os.path.dirname(__file__), "documents")

    @property
    def masked_database_url(self) -> str:
        """Return a redacted database URL for logs and status output."""
        parsed = urlsplit(self.database_url)
        if parsed.password is None:
            return self.database_url

        userinfo = parsed.username or ""
        if userinfo:
            userinfo = f"{userinfo}:***"
        else:
            userinfo = "***"

        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"

        return urlunsplit((parsed.scheme, f"{userinfo}@{host}", parsed.path, parsed.query, parsed.fragment))
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if not self.database_url:
            raise ValueError("DATABASE_URL must be set")
        if not self.vector_table:
            raise ValueError("VECTOR_TABLE must be set")
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return True


def get_config() -> IngestionConfig:
    """Get the ingestion configuration."""
    config = IngestionConfig()
    config.validate()
    return config

