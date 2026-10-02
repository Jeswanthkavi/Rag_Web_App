"""Semantic retrieval service coordinating query validation, embedding, and tenant-isolated vector search."""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from backend.app.embeddings.base import EmbeddingProviderError
from backend.app.embeddings.service import EmbeddingService
from backend.app.retrieval.config import RetrievalConfig
from backend.app.retrieval.models import (
    RetrievalError,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
    RetrievalValidationError,
)
from backend.app.vectorstore.client import create_qdrant_client
from backend.app.vectorstore.models import VectorStoreError
from backend.app.vectorstore.repository import QdrantVectorRepository

logger = logging.getLogger("app.retrieval.service")


class RetrievalService:
    """Production retrieval service executing tenant-scoped semantic similarity search."""

    def __init__(
        self,
        vector_repo: Optional[QdrantVectorRepository] = None,
        embedding_service: Optional[EmbeddingService] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.config = config or RetrievalConfig.from_settings()
        self.embedding_service = embedding_service or EmbeddingService()

        if vector_repo:
            self.vector_repo = vector_repo
        else:
            client = create_qdrant_client()
            self.vector_repo = QdrantVectorRepository(client=client)

    def validate_query(self, query: str) -> str:
        """Validates and sanitizes a natural language search query."""
        if query is None:
            raise RetrievalValidationError("Query cannot be None.")
        
        trimmed = query.strip()
        if not trimmed:
            raise RetrievalValidationError("Query string cannot be empty or whitespace-only.")
        
        if len(trimmed) > self.config.max_query_length:
            raise RetrievalValidationError(
                f"Query length ({len(trimmed)}) exceeds the maximum allowed length of {self.config.max_query_length} characters."
            )
        
        return trimmed

    def search(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> RetrievalResponse:
        """Executes tenant-isolated semantic retrieval.
        
        Workflow:
        1. Validate query string and tenant user_id.
        2. Resolve effective top-K and similarity threshold parameters.
        3. Generate 384-dimensional dense vector for the query via EmbeddingService.
        4. Execute Qdrant vector search with mandatory tenant scoping (and optional document scoping).
        5. Validate payload structures and filter points by similarity threshold.
        6. Order by descending score and assign 1-indexed ranks.
        7. Return structured RetrievalResponse.
        """
        # 1. Validation
        sanitized_query = self.validate_query(query)

        if not user_id or not user_id.strip():
            raise RetrievalValidationError("user_id is required for tenant-isolated retrieval.")
        clean_user_id = user_id.strip()
        clean_doc_id = document_id.strip() if document_id and document_id.strip() else None

        # 2. Parameter resolution
        effective_top_k = top_k if top_k is not None else self.config.top_k
        if effective_top_k < 1:
            raise RetrievalValidationError(f"top_k must be at least 1, got {effective_top_k}")
        if effective_top_k > 100:
            raise RetrievalValidationError(f"top_k cannot exceed 100, got {effective_top_k}")

        effective_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.config.similarity_threshold
        )
        if effective_threshold is not None:
            if not (0.0 <= effective_threshold <= 1.0):
                raise RetrievalValidationError(
                    f"similarity_threshold must be between 0.0 and 1.0, got {effective_threshold}"
                )

        target_collection = collection_name or self.config.collection_name

        # 3. Generate query embedding
        try:
            query_vector = self.embedding_service.embed_text(sanitized_query)
        except (EmbeddingProviderError, Exception) as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Query embedding generation failed: {sanitized}")
            raise RetrievalError(f"Query embedding failed: {sanitized}") from e

        # 4. Execute Qdrant semantic search
        try:
            raw_matches = self.vector_repo.search(
                query_vector=query_vector,
                user_id=clean_user_id,
                document_id=clean_doc_id,
                top_k=effective_top_k,
                score_threshold=effective_threshold,
                collection_name=target_collection,
            )
        except (VectorStoreError, Exception) as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Qdrant search failed for user '{clean_user_id}': {sanitized}")
            raise RetrievalError(f"Semantic search failed: {sanitized}") from e

        # 5. Payload validation & score threshold filtering
        valid_results: List[RetrievalResult] = []
        for match in raw_matches:
            score = match.get("score", 0.0)

            # Application-level threshold verification
            if effective_threshold is not None and score < effective_threshold:
                continue

            payload: Dict[str, Any] = match.get("payload") or {}
            point_id = match.get("id", "")

            # Validate required payload fields
            chunk_id = payload.get("chunk_id") or point_id
            doc_id = payload.get("document_id")
            tenant_id = payload.get("user_id")
            chunk_text = payload.get("chunk_text")
            chunk_idx = payload.get("chunk_index")
            filename = payload.get("filename")
            file_type = payload.get("file_type")

            if not chunk_id or not doc_id or not tenant_id or chunk_text is None:
                logger.warning(
                    f"Skipping malformed vector point '{point_id}': missing essential payload fields (doc_id={doc_id}, user_id={tenant_id})"
                )
                continue

            # Tenant integrity check on payload
            if str(tenant_id) != clean_user_id:
                logger.error(
                    f"Security anomaly: Qdrant returned point '{point_id}' with payload user_id '{tenant_id}' differing from query user_id '{clean_user_id}'"
                )
                continue

            result_item = RetrievalResult(
                chunk_id=str(chunk_id),
                document_id=str(doc_id),
                user_id=str(tenant_id),
                chunk_index=int(chunk_idx) if chunk_idx is not None else 0,
                chunk_text=str(chunk_text),
                page_number=payload.get("page_number"),
                filename=str(filename) if filename else "unknown",
                file_type=str(file_type) if file_type else "unknown",
                token_count=payload.get("token_count"),
                score=float(score),
                rank=1,  # Rank will be assigned after final sorting
            )
            valid_results.append(result_item)

        # 6. Order by descending score and assign 1-indexed ranks
        valid_results.sort(key=lambda x: x.score, reverse=True)
        ranked_results: List[RetrievalResult] = []
        for rank_idx, item in enumerate(valid_results[:effective_top_k], start=1):
            item.rank = rank_idx
            ranked_results.append(item)

        logger.info(
            f"Retrieval complete for user '{clean_user_id}': query='{sanitized_query[:30]}...' -> {len(ranked_results)} matches (top_k={effective_top_k}, threshold={effective_threshold})"
        )

        # 7. Construct structured response
        return RetrievalResponse(
            query=sanitized_query,
            results=ranked_results,
            total_retrieved=len(ranked_results),
            top_k=effective_top_k,
            similarity_threshold=effective_threshold,
        )
