"""Base abstractions and exceptions for candidate chunk reranking."""

from abc import ABC, abstractmethod
from typing import List, Optional
from backend.app.retrieval.models import RetrievalResult
from backend.app.reranking.models import RerankResult


class RerankerError(Exception):
    """Base exception for reranker provider errors."""
    pass


class BaseReranker(ABC):
    """Abstract base class for chunk reranking providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier name of the reranker provider."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Reranks candidate retrieval chunks against a query using a cross-encoder model.
        
        Args:
            query: The search or retrieval query string.
            candidates: List of candidate RetrievalResult chunks from dense retrieval.
            top_k: Optional limit on the number of top candidates to return.
            
        Returns:
            List of RerankResult instances sorted descending by rerank_score.
        """
        pass
