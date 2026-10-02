"""Domain models for query rewriting requests and responses."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.app.query_rewriting.base import RewrittenQuery


class QueryRewritingRequest(BaseModel):
    """Input payload for query rewriting."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=1000, description="The current user query to rewrite")
    conversation_history: Optional[str] = Field(default=None, description="Prior conversation turns formatted as text")


__all__ = ["RewrittenQuery", "QueryRewritingRequest"]
