"""Qdrant client initialization and lifecycle management."""

import logging
import threading
from typing import Dict, Optional, Tuple
from qdrant_client import QdrantClient
from backend.app.vectorstore.config import VectorStoreConfig

logger = logging.getLogger("app.vectorstore.client")

_CLIENT_CACHE: Dict[Tuple[str, Optional[str], bool], QdrantClient] = {}
_CLIENT_LOCK = threading.Lock()


def create_qdrant_client(config: Optional[VectorStoreConfig] = None) -> QdrantClient:
    """Creates or retrieves a cached configured instance of QdrantClient."""
    cfg = config or VectorStoreConfig.from_settings()
    key = (cfg.url, cfg.api_key, cfg.prefer_grpc)
    with _CLIENT_LOCK:
        if key not in _CLIENT_CACHE:
            logger.debug(f"Initializing Qdrant client connection for URL: {cfg.url}")
            if cfg.url == ":memory:":
                _CLIENT_CACHE[key] = QdrantClient(location=":memory:")
            else:
                _CLIENT_CACHE[key] = QdrantClient(
                    url=cfg.url,
                    api_key=cfg.api_key,
                    timeout=cfg.timeout,
                    prefer_grpc=cfg.prefer_grpc,
                )
        return _CLIENT_CACHE[key]
