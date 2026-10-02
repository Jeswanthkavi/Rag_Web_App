"""Base abstractions and exceptions for lexical and hybrid retrieval."""

from abc import ABC, abstractmethod
from typing import List, Optional
from backend.app.retrieval.models import RetrievalResult


class LexicalRetrievalError(Exception):
    """Base exception for lexical retrieval errors."""
    pass


class HybridRetrievalError(Exception):
    """Base exception for hybrid retrieval errors."""
    pass


class BaseLexicalRetriever(ABC):
    """Abstract base class for lexical/sparse retrieval providers (e.g. BM25)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier name of the lexical provider."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Performs tenant-isolated lexical keyword/BM25 retrieval over document chunks.
        
        Args:
            query: Natural language keyword search query.
            user_id: Tenant user ID strictly scoping the search.
            top_k: Optional maximum number of lexical matches to return.
            document_id: Optional document ID scoping filter.
            
        Returns:
            List of RetrievalResult objects ranked descending by lexical score.
        """
        pass
