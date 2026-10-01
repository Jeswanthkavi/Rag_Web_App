"""Chunking configuration definitions and settings resolution."""

from dataclasses import dataclass
from typing import List, Optional
from backend.app.core.config import get_settings


@dataclass
class ChunkingConfig:
    """Configuration parameters for document text splitting."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: Optional[List[str]] = None

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

    @classmethod
    def from_settings(cls) -> "ChunkingConfig":
        """Loads chunking configuration directly from global application settings."""
        settings = get_settings()
        return cls(
            chunk_size=settings.retrieval.chunk_size,
            chunk_overlap=settings.retrieval.chunk_overlap,
        )
