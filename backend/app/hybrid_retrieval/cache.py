"""In-process, thread-safe, tenant-isolated BM25 corpus and index cache."""

import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from backend.app.hybrid_retrieval.bm25 import BM25Okapi

logger = logging.getLogger("app.hybrid_retrieval.cache")


class TenantBM25Cache:
    """Thread-safe bounded cache for pre-tokenized corpora and BM25 index models.
    
    Guarantees strict tenant isolation:
    - Cache key is strictly (user_id, document_id) tuple.
    - Validates corpus fingerprint (chunk_count, min_chunk_id, max_chunk_id) on retrieval.
    - Prevents cross-tenant data access.
    """

    def __init__(self, max_entries: int = 100):
        self._max_entries = max_entries
        self._cache: OrderedDict[Tuple[str, Optional[str]], Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(self, user_id: str, document_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
        clean_user = user_id.strip() if user_id else ""
        clean_doc = document_id.strip() if document_id and document_id.strip() else None
        return (clean_user, clean_doc)

    def get(
        self,
        user_id: str,
        document_id: Optional[str],
        expected_fingerprint: Tuple[int, Optional[str], Optional[str]],
    ) -> Optional[Tuple[BM25Okapi, List[Tuple[Any, str, str]]]]:
        """Retrieves cached (bm25_model, chunks) if fingerprint matches."""
        key = self._make_key(user_id, document_id)
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.get("fingerprint") != expected_fingerprint:
                # Fingerprint mismatch (chunks added/removed/updated) -> invalidate
                del self._cache[key]
                logger.debug(f"BM25 cache fingerprint mismatch for tenant '{user_id}', invalidated.")
                return None

            # Move to end (MRU)
            self._cache.move_to_end(key)
            return entry["bm25"], entry["chunks"]

    def put(
        self,
        user_id: str,
        document_id: Optional[str],
        fingerprint: Tuple[int, Optional[str], Optional[str]],
        bm25: BM25Okapi,
        chunks: List[Tuple[Any, str, str]],
    ) -> None:
        """Stores BM25 model and chunk metadata in tenant-scoped cache."""
        key = self._make_key(user_id, document_id)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = {
                "fingerprint": fingerprint,
                "bm25": bm25,
                "chunks": chunks,
            }
            # Evict LRU entries if exceeded max_entries
            while len(self._cache) > self._max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Evicted BM25 cache entry for tenant key: {evicted_key}")

    def invalidate_user(self, user_id: str) -> None:
        """Invalidates all cached BM25 entries for a specific tenant user."""
        clean_user = user_id.strip() if user_id else ""
        with self._lock:
            keys_to_delete = [k for k in self._cache if k[0] == clean_user]
            for k in keys_to_delete:
                del self._cache[k]
            if keys_to_delete:
                logger.debug(f"Invalidated {len(keys_to_delete)} BM25 cache entries for user '{clean_user}'")

    def invalidate_document(self, user_id: str, document_id: str) -> None:
        """Invalidates cached BM25 entries for a specific document and tenant."""
        clean_user = user_id.strip() if user_id else ""
        clean_doc = document_id.strip() if document_id else ""
        with self._lock:
            keys_to_delete = [
                k for k in self._cache
                if k[0] == clean_user and (k[1] == clean_doc or k[1] is None)
            ]
            for k in keys_to_delete:
                del self._cache[k]

    def clear(self) -> None:
        """Clears entire BM25 cache."""
        with self._lock:
            self._cache.clear()


# Global thread-safe singleton cache
_global_bm25_cache = TenantBM25Cache()


def get_bm25_cache() -> TenantBM25Cache:
    """Returns the global TenantBM25Cache singleton."""
    return _global_bm25_cache
