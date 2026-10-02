"""Retrieval configuration and settings loader."""

from dataclasses import dataclass
from typing import Optional
from backend.app.core.config import Settings, get_settings


@dataclass(frozen=True)
class RetrievalConfig:
    """Runtime configuration for semantic retrieval."""
    top_k: int = 5
    similarity_threshold: Optional[float] = 0.5
    max_query_length: int = 1000
    collection_name: Optional[str] = None

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if self.top_k > 100:
            raise ValueError(f"top_k cannot exceed 100, got {self.top_k}")
        if self.similarity_threshold is not None:
            if not (0.0 <= self.similarity_threshold <= 1.0):
                raise ValueError(
                    f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
                )
        if self.max_query_length < 10 or self.max_query_length > 10000:
            raise ValueError(
                f"max_query_length must be between 10 and 10000, got {self.max_query_length}"
            )

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> "RetrievalConfig":
        """Builds RetrievalConfig from application Pydantic settings with optional overrides."""
        s = settings or get_settings()
        k = top_k if top_k is not None else s.retrieval.retrieval_top_k
        thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else s.retrieval.score_threshold
        )
        coll = collection_name or s.qdrant.qdrant_collection_name
        return cls(
            top_k=k,
            similarity_threshold=thresh,
            collection_name=coll,
        )
