"""Data models for candidate reranking results and responses."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.app.retrieval.models import RetrievalResult


class RerankResult(BaseModel):
    """A reranked chunk candidate with dense score, rerank score, and rank."""
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
    dense_score: float = Field(..., description="Original cosine similarity score from dense vector search")
    rerank_score: float = Field(..., description="Relevance score assigned by the cross-encoder reranker")
    rank: int = Field(..., ge=1, description="1-indexed rank position after reranking")

    @classmethod
    def from_retrieval_result(
        cls,
        result: RetrievalResult,
        rerank_score: float,
        rank: int,
    ) -> "RerankResult":
        """Constructs a RerankResult from an existing RetrievalResult."""
        return cls(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            user_id=result.user_id,
            chunk_index=result.chunk_index,
            chunk_text=result.chunk_text,
            page_number=result.page_number,
            filename=result.filename,
            file_type=result.file_type,
            token_count=result.token_count,
            dense_score=result.score,
            rerank_score=rerank_score,
            rank=rank,
        )


class RerankResponse(BaseModel):
    """Structured response containing ranked results and reranking metadata."""
    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., description="The query used for scoring candidates")
    results: List[RerankResult] = Field(default_factory=list, description="Ranked chunks sorted by rerank_score")
    total_candidates: int = Field(..., ge=0, description="Total candidates evaluated before top-K slicing")
    final_top_k: int = Field(..., ge=1, description="Top-K limit applied to final reranked results")
