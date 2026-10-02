"""Hybrid retrieval service orchestrating parallel dense and lexical retrieval with RRF fusion."""

import logging
import time
from typing import List, Optional

from backend.app.hybrid_retrieval.base import (
    BaseLexicalRetriever,
    HybridRetrievalError,
    LexicalRetrievalError,
)
from backend.app.hybrid_retrieval.config import HybridRetrievalConfig
from backend.app.hybrid_retrieval.fusion import reciprocal_rank_fusion
from backend.app.hybrid_retrieval.lexical_provider import SQLLexicalRetriever
from backend.app.hybrid_retrieval.models import HybridRetrievalResponse
from backend.app.retrieval.models import (
    RetrievalError,
    RetrievalResponse,
    RetrievalResult,
    RetrievalValidationError,
)
from backend.app.retrieval.service import RetrievalService

logger = logging.getLogger("app.hybrid_retrieval.service")


class HybridRetrievalService:
    """Coordinates dense semantic vector search and lexical BM25 search with Reciprocal Rank Fusion."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        lexical_retriever: Optional[BaseLexicalRetriever] = None,
        config: Optional[HybridRetrievalConfig] = None,
    ):
        self.config = config or HybridRetrievalConfig.from_settings()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.lexical_retriever = lexical_retriever or SQLLexicalRetriever(config=self.config)

    def search(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> HybridRetrievalResponse:
        """Executes hybrid retrieval combining dense vector search and lexical search with RRF fusion.
        
        Workflow:
        1. If hybrid retrieval is disabled, delegates directly to dense RetrievalService.
        2. Executes tenant-isolated dense retrieval (fetching dense_top_k candidates).
        3. Executes tenant-isolated lexical retrieval (fetching lexical_top_k candidates).
        4. Applies Reciprocal Rank Fusion (RRF) and deduplication.
        5. Returns top fused candidates up to fusion_candidate_limit (or top_k override).
        
        Fallback:
        - If dense fails but lexical succeeds: returns lexical candidates.
        - If lexical fails but dense succeeds: returns dense candidates.
        - If both fail: raises HybridRetrievalError.
        """
        if not user_id or not user_id.strip():
            raise RetrievalValidationError("user_id is required for hybrid retrieval.")

        if not query or not query.strip():
            raise RetrievalValidationError("Query cannot be empty or whitespace-only.")

        clean_user_id = user_id.strip()
        clean_doc_id = document_id.strip() if document_id and document_id.strip() else None
        limit = top_k if top_k is not None else self.config.fusion_candidate_limit

        # 1. Disabled bypass
        if not self.config.enabled:
            logger.debug("Hybrid retrieval is disabled; delegating directly to dense retrieval.")
            dense_resp = self.retrieval_service.search(
                query=query,
                user_id=clean_user_id,
                top_k=limit,
                similarity_threshold=similarity_threshold,
                document_id=clean_doc_id,
            )
            return HybridRetrievalResponse(
                query=query,
                results=dense_resp.results,
                candidates=[],
                total_fused=len(dense_resp.results),
                dense_count=len(dense_resp.results),
                lexical_count=0,
            )

        # 2. Execute Dense Retrieval Branch
        dense_results: List[RetrievalResult] = []
        dense_failed = False
        dense_k = self.config.dense_top_k if top_k is None else max(top_k, self.config.dense_top_k)

        try:
            start_dense = time.perf_counter()
            dense_resp = self.retrieval_service.search(
                query=query,
                user_id=clean_user_id,
                top_k=dense_k,
                similarity_threshold=similarity_threshold,
                document_id=clean_doc_id,
            )
            dense_results = dense_resp.results
            dense_duration = time.perf_counter() - start_dense
            logger.debug(f"Dense branch retrieved {len(dense_results)} chunks in {dense_duration:.3f}s")
        except Exception as e:
            dense_failed = True
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.warning(f"Dense retrieval branch failed: {sanitized}")

        # 3. Execute Lexical Retrieval Branch
        lexical_results: List[RetrievalResult] = []
        lexical_failed = False
        lex_k = self.config.lexical_top_k if top_k is None else max(top_k, self.config.lexical_top_k)

        try:
            start_lex = time.perf_counter()
            lexical_results = self.lexical_retriever.search(
                query=query,
                user_id=clean_user_id,
                top_k=lex_k,
                document_id=clean_doc_id,
            )
            lex_duration = time.perf_counter() - start_lex
            logger.debug(f"Lexical branch retrieved {len(lexical_results)} chunks in {lex_duration:.3f}s")
        except Exception as e:
            lexical_failed = True
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.warning(f"Lexical retrieval branch failed: {sanitized}")

        # 4. Handle Branch Failures & Fusion
        if dense_failed and lexical_failed:
            raise HybridRetrievalError("Both dense and lexical retrieval branches failed.")

        if dense_failed:
            logger.warning("Falling back exclusively to lexical retrieval results.")
            sliced_lex = lexical_results[:limit]
            return HybridRetrievalResponse(
                query=query,
                results=sliced_lex,
                candidates=[],
                total_fused=len(sliced_lex),
                dense_count=0,
                lexical_count=len(lexical_results),
            )

        if lexical_failed or not lexical_results:
            if lexical_failed:
                logger.warning("Falling back exclusively to dense retrieval results.")
            sliced_dense = dense_results[:limit]
            return HybridRetrievalResponse(
                query=query,
                results=sliced_dense,
                candidates=[],
                total_fused=len(sliced_dense),
                dense_count=len(dense_results),
                lexical_count=0,
            )

        # 5. Apply Reciprocal Rank Fusion (RRF)
        fused_results, candidates = reciprocal_rank_fusion(
            dense_results=dense_results,
            lexical_results=lexical_results,
            rrf_k=self.config.rrf_k,
            limit=limit,
        )

        return HybridRetrievalResponse(
            query=query,
            results=fused_results,
            candidates=candidates,
            total_fused=len(fused_results),
            dense_count=len(dense_results),
            lexical_count=len(lexical_results),
        )
