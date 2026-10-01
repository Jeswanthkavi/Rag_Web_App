"""Vector store configuration and factory methods."""

from dataclasses import dataclass
from typing import Optional
from backend.app.core.config import get_settings


@dataclass
class VectorStoreConfig:
    """Configuration settings for Qdrant vector database connection and collections."""
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    collection_name: str = "rag_document_chunks"
    timeout: float = 10.0
    batch_size: int = 64
    prefer_grpc: bool = False
    dimension: int = 384
    distance: str = "Cosine"

    def __post_init__(self):
        if not self.url or not self.url.strip():
            raise ValueError("Qdrant URL cannot be empty.")
        if not self.collection_name or not self.collection_name.strip():
            raise ValueError("Qdrant collection_name cannot be empty.")
        if self.timeout <= 0:
            raise ValueError(f"Qdrant timeout must be positive, got {self.timeout}")
        if self.batch_size <= 0:
            raise ValueError(f"Qdrant batch_size must be positive, got {self.batch_size}")
        if self.dimension <= 0:
            raise ValueError(f"Vector dimension must be positive, got {self.dimension}")

    @classmethod
    def from_settings(cls, collection_name: Optional[str] = None) -> "VectorStoreConfig":
        """Instantiates VectorStoreConfig directly from application settings."""
        settings = get_settings()
        api_key_str = settings.qdrant.qdrant_api_key.get_secret_value() if settings.qdrant.qdrant_api_key else None
        return cls(
            url=settings.qdrant.qdrant_url,
            api_key=api_key_str,
            collection_name=collection_name or settings.qdrant.qdrant_collection_name,
            timeout=settings.qdrant.qdrant_timeout,
            batch_size=settings.qdrant.qdrant_batch_size,
            prefer_grpc=settings.qdrant.prefer_grpc,
            dimension=settings.embedding.embedding_dimension,
        )
