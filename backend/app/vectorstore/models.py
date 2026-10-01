"""Data transfer models and exceptions for vector store operations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class VectorStoreError(Exception):
    """Base exception for all vector store and collection operations."""
    pass


@dataclass
class VectorPoint:
    """Represents a single vector point ready for upsert into Qdrant."""
    id: str  # Deterministic UUID string matching DocumentChunk.id
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexingResult:
    """Summary result returned after indexing a document's chunks in Qdrant."""
    document_id: str
    user_id: str
    indexed_chunks_count: int
    collection_name: str
    point_ids: List[str] = field(default_factory=list)


@dataclass
class VectorStoreHealth:
    """Health check diagnostic data for the vector store instance."""
    status: str
    collection_name: str
    collection_exists: bool
    points_count: int
    dimension: int
    distance: str
