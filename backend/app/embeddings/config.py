"""Embedding configuration dataclass and factory."""

from dataclasses import dataclass
from typing import Optional
from backend.app.core.config import get_settings


@dataclass
class EmbeddingConfig:
    """Configuration parameters for dense embedding generation."""
    provider: str = "local"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize_embeddings: bool = True

    def __post_init__(self):
        if not self.provider or not self.provider.strip():
            raise ValueError("Embedding provider cannot be empty.")
        if not self.model_name or not self.model_name.strip():
            raise ValueError("Embedding model_name cannot be empty.")
        if self.dimension <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {self.dimension}")
        if self.batch_size <= 0:
            raise ValueError(f"Embedding batch_size must be positive, got {self.batch_size}")

    @classmethod
    def from_settings(cls) -> "EmbeddingConfig":
        """Builds an EmbeddingConfig instance from system-level application settings."""
        settings = get_settings()
        return cls(
            provider=settings.embedding.embedding_provider,
            model_name=settings.embedding.embedding_model_name,
            dimension=settings.embedding.embedding_dimension,
            batch_size=settings.embedding.embedding_batch_size,
            normalize_embeddings=settings.embedding.normalize_embeddings,
        )
