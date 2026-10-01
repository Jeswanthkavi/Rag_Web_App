"""Chunking package providing deterministic text splitting and chunk management."""

from backend.app.chunking.config import ChunkingConfig
from backend.app.chunking.splitter import (
    ChunkData,
    count_tokens_approx,
    RecursiveCharacterSplitter,
)
from backend.app.chunking.service import ChunkingService

__all__ = [
    "ChunkingConfig",
    "ChunkData",
    "count_tokens_approx",
    "RecursiveCharacterSplitter",
    "ChunkingService",
]
