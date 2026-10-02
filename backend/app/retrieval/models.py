"""Data models and exceptions for semantic retrieval and similarity search."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class RetrievalError(Exception):
    """Base exception for retrieval-layer operations."""
    pass


class RetrievalValidationError(RetrievalError):
    """Exception raised when query parameters fail validation."""
    pass


class RetrievalQuery(BaseModel):
    """Validated semantic search query request."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=1000, description="Natural language search query")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Number of results to retrieve")
    similarity_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum cosine similarity score threshold"
    )
    document_id: Optional[str] = Field(default=None, description="Optional document ID scoping filter")


class RetrievalResult(BaseModel):
    """A single retrieved chunk match with semantic score and citation metadata."""
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str = Field(..., description="Unique UUID identifier of the document chunk")
    document_id: str = Field(..., description="UUID identifier of the parent document")
    user_id: str = Field(..., description="Tenant user ID owning this chunk")
    chunk_index: int = Field(..., description="0-indexed position within the document")
    chunk_text: str = Field(..., description="Extracted textual content of the chunk")
    page_number: Optional[int] = Field(default=None, description="Original source document page number if paged")
    filename: str = Field(..., description="Original filename of the parent document")
    file_type: str = Field(..., description="File extension / type (.pdf, .docx, .txt)")
    token_count: Optional[int] = Field(default=None, description="Approximate token count of the chunk text")
    score: float = Field(..., description="Cosine similarity score from vector search [0.0 - 1.0]")
    rank: int = Field(..., ge=1, description="1-indexed rank ordering among retrieved results")


class RetrievalResponse(BaseModel):
    """Structured response containing ranked retrieval results and search metadata."""
    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., description="The validated and processed search query string")
    results: List[RetrievalResult] = Field(default_factory=list, description="Ranked list of retrieved chunk results")
    total_retrieved: int = Field(..., ge=0, description="Total number of chunks returned after filtering")
    top_k: int = Field(..., ge=1, description="Top-K limit applied to the query")
    similarity_threshold: Optional[float] = Field(
        default=None, description="Similarity threshold applied to filter results"
    )
