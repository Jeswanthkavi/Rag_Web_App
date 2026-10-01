"""Embedding service coordinating provider selection, text embedding, and chunk batching."""

import logging
from typing import Dict, List, Optional, Type, TYPE_CHECKING

from backend.app.embeddings.base import BaseEmbeddingProvider, EmbeddingProviderError
from backend.app.embeddings.config import EmbeddingConfig
from backend.app.embeddings.local_provider import LocalEmbeddingProvider
from backend.app.embeddings.models import ChunkEmbedding, EmbeddingMetadata

if TYPE_CHECKING:
    from backend.app.models.document_chunk import DocumentChunk

logger = logging.getLogger("app.embeddings.service")


class EmbeddingService:
    """High-level service orchestrating dense vector generation across configurable providers."""

    _PROVIDER_REGISTRY: Dict[str, Type[BaseEmbeddingProvider]] = {
        "local": LocalEmbeddingProvider,
    }

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        provider: Optional[BaseEmbeddingProvider] = None,
    ):
        self.config = config or EmbeddingConfig.from_settings()
        if provider:
            self.provider = provider
        else:
            self.provider = self._resolve_provider(self.config)

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseEmbeddingProvider]) -> None:
        """Registers a new provider implementation."""
        cls._PROVIDER_REGISTRY[name.lower()] = provider_cls

    def _resolve_provider(self, config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """Instantiates the configured embedding provider from the registry."""
        provider_key = config.provider.lower()
        provider_cls = self._PROVIDER_REGISTRY.get(provider_key)
        if not provider_cls:
            raise EmbeddingProviderError(
                f"Unsupported embedding provider: '{config.provider}'. Available: {list(self._PROVIDER_REGISTRY.keys())}"
            )
        return provider_cls(config)

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def get_metadata(self) -> EmbeddingMetadata:
        """Returns structured metadata about the active embedding configuration."""
        return EmbeddingMetadata(
            provider=self.provider_name,
            model_name=self.model_name,
            dimension=self.dimension,
            normalize_embeddings=self.config.normalize_embeddings,
        )

    def embed_text(self, text: str) -> List[float]:
        """Generates a dense embedding vector for a single query or text snippet."""
        return self.provider.embed_text(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates dense embedding vectors for a batch of text strings."""
        if not texts:
            return []
        return self.provider.embed_texts(texts)

    def embed_chunks(self, chunks: List["DocumentChunk"]) -> List[ChunkEmbedding]:
        """Generates embeddings for a list of persisted DocumentChunk records.
        
        Preserves strict positional correspondence and metadata association.
        """
        if not chunks:
            return []

        texts = [c.chunk_text for c in chunks]
        vectors = self.embed_texts(texts)

        if len(vectors) != len(chunks):
            raise EmbeddingProviderError(
                f"Mismatch between chunks count ({len(chunks)}) and generated vectors ({len(vectors)})"
            )

        results: List[ChunkEmbedding] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            results.append(
                ChunkEmbedding(
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index if chunk.chunk_index is not None else idx,
                    document_id=chunk.document_id,
                    vector=vector,
                    dimension=len(vector),
                    token_count=chunk.token_count,
                    model_name=self.model_name,
                )
            )

        return results
