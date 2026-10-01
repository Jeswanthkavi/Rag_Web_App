"""Data models and transfer objects for embedding operations."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkEmbedding:
    """Represents a generated dense embedding vector associated with a DocumentChunk."""
    chunk_id: Optional[str]
    chunk_index: int
    document_id: Optional[str]
    vector: List[float]
    dimension: int
    token_count: Optional[int] = None
    model_name: Optional[str] = None


@dataclass
class EmbeddingMetadata:
    """Metadata describing an active embedding provider and configuration."""
    provider: str
    model_name: str
    dimension: int
    normalize_embeddings: bool
