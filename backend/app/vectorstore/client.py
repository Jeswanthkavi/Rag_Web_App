"""Qdrant client initialization and lifecycle management."""

import logging
from typing import Optional
from qdrant_client import QdrantClient
from backend.app.vectorstore.config import VectorStoreConfig

logger = logging.getLogger("app.vectorstore.client")


def create_qdrant_client(config: Optional[VectorStoreConfig] = None) -> QdrantClient:
    """Creates a configured instance of QdrantClient."""
    cfg = config or VectorStoreConfig.from_settings()
    logger.debug(f"Initializing Qdrant client for URL: {cfg.url}")
    return QdrantClient(
        url=cfg.url,
        api_key=cfg.api_key,
        timeout=cfg.timeout,
        prefer_grpc=cfg.prefer_grpc,
    )
