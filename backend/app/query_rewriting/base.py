"""Base abstractions and data contracts for conversational query rewriting."""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class QueryRewriterError(Exception):
    """Base exception for query rewriter provider errors."""
    pass


class RewrittenQuery(BaseModel):
    """Structured representation of query rewriting result."""
    model_config = ConfigDict(from_attributes=True)

    original_query: str = Field(..., description="The original unmodified user query")
    retrieval_query: str = Field(..., description="The standalone rewritten query to be used for retrieval")
    was_rewritten: bool = Field(default=False, description="Whether the query was modified during rewriting")
    reason: Optional[str] = Field(default=None, description="Diagnostic reason or strategy applied during rewriting")


class BaseQueryRewriter(ABC):
    """Abstract base class for query rewriting providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier name of the rewriter provider."""
        pass

    @abstractmethod
    def rewrite(
        self,
        query: str,
        conversation_history: Optional[str] = None,
    ) -> RewrittenQuery:
        """Transforms a conversational follow-up query into a standalone retrieval query.
        
        Args:
            query: The current user query.
            conversation_history: Formatted bounded history of prior conversation turns.
            
        Returns:
            RewrittenQuery containing the original and retrieval queries.
        """
        pass
