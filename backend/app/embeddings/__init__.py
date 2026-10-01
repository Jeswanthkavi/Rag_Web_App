"""Embedding package providing provider abstraction, local inference, and service orchestration."""

from backend.app.embeddings.base import BaseEmbeddingProvider, EmbeddingProviderError
from backend.app.embeddings.config import EmbeddingConfig
from backend.app.embeddings.local_provider import LocalEmbeddingProvider
from backend.app.embeddings.models import ChunkEmbedding, EmbeddingMetadata
from backend.app.embeddings.service import EmbeddingService

__all__ = [
    "BaseEmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingConfig",
    "LocalEmbeddingProvider",
    "ChunkEmbedding",
    "EmbeddingMetadata",
    "EmbeddingService",
]
