"""Reranking service coordinating candidate reordering, top-K slicing, and fallback handling."""

import logging
from typing import List, Optional

from backend.app.reranking.base import BaseReranker, RerankerError
from backend.app.reranking.config import RerankingConfig
from backend.app.reranking.local_provider import LocalReranker
from backend.app.reranking.models import RerankResult
from backend.app.retrieval.models import RetrievalResult

logger = logging.getLogger("app.reranking.service")


class RerankingService:
    """Service orchestrating candidate chunk reranking with safe fallback to dense retrieval."""

    def __init__(
        self,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RerankingConfig] = None,
    ):
        self.config = config or RerankingConfig.from_settings()
        self.reranker = reranker or LocalReranker(config=self.config)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Reranks dense candidate chunks and returns top-K scored chunks.
        
        Guarantees:
        1. Tenant Safety: Works exclusively on the already tenant-filtered candidates passed in.
        2. Score Distinction: Preserves dense_score and rerank_score as separate attributes.
        3. Safe Fallback: If reranking is disabled or if the reranker provider fails, gracefully
           falls back to dense retrieval ordering without failing the RAG request.
        4. Top-K Slicing: Returns at most top_k (or config.final_top_k) candidates.
        """
        if not candidates:
            return []

        limit = top_k if top_k is not None else self.config.final_top_k

        # 1. Check if reranking is enabled
        if not self.config.enabled:
            logger.debug("Reranking is disabled; preserving dense retrieval candidate order.")
            sliced = candidates[:limit]
            return [
                RerankResult.from_retrieval_result(
                    result=c,
                    rerank_score=c.score,
                    rank=idx + 1,
                )
                for idx, c in enumerate(sliced)
            ]

        # 2. Execute reranker with fallback protection
        try:
            reranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=limit,
            )
            return reranked

        except Exception as e:
            err_type = type(e).__name__
            logger.warning(
                f"Reranking provider failed ({err_type}). Falling back to dense retrieval ranking."
            )
            sliced = candidates[:limit]
            return [
                RerankResult.from_retrieval_result(
                    result=c,
                    rerank_score=c.score,
                    rank=idx + 1,
                )
                for idx, c in enumerate(sliced)
            ]
