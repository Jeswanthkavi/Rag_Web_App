from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from backend.app.citations.models import Citation


class RAGError(Exception):
    """Base exception for RAG answer generation errors."""
    pass


class RAGValidationError(RAGError):
    """Exception raised when RAG query parameters fail validation."""
    pass


class RAGQuery(BaseModel):
    """User request for grounded RAG answer generation."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=1000, description="Natural language user question")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Maximum number of context chunks to retrieve")
    similarity_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum cosine similarity score threshold"
    )
    document_id: Optional[str] = Field(default=None, description="Optional document ID scoping filter")
    conversation_id: Optional[str] = Field(default=None, description="Optional conversation session ID for multi-turn context")


class RAGResponse(BaseModel):
    """Structured response containing generated grounded answer, citations, and conversation metadata."""
    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., description="The validated natural language query")
    answer: str = Field(..., description="The grounded answer synthesized from retrieved document context")
    retrieved_context_count: int = Field(..., ge=0, description="Number of context chunks retrieved and used for generation")
    citations: List[Citation] = Field(default_factory=list, description="Authoritative verified citations for claims made in the answer")
    conversation_id: Optional[str] = Field(default=None, description="Associated conversation thread ID")
    message_id: Optional[str] = Field(default=None, description="Unique ID of the generated assistant message record")
    model_name: Optional[str] = Field(default=None, description="LLM model identifier used for generation")
