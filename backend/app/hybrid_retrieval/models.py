"""Domain models for hybrid candidate ranking and fusion results."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.app.retrieval.models import RetrievalResult


class HybridCandidate(BaseModel):
    """A hybrid retrieval candidate retaining dense, lexical, and fusion score telemetry."""
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
    dense_score: Optional[float] = Field(default=None, description="Cosine similarity score from vector search if retrieved")
    dense_rank: Optional[int] = Field(default=None, description="1-indexed rank in dense retrieval results if retrieved")
    lexical_score: Optional[float] = Field(default=None, description="BM25/lexical score if retrieved")
    lexical_rank: Optional[int] = Field(default=None, description="1-indexed rank in lexical retrieval results if retrieved")
    fusion_score: float = Field(default=0.0, description="Combined Reciprocal Rank Fusion (RRF) score")
    rank: int = Field(default=1, ge=1, description="1-indexed final rank in the fused candidate set")

    def to_retrieval_result(self) -> RetrievalResult:
        """Converts to a standard RetrievalResult instance compatible with rerankers and context builders."""
        # Use fusion_score or dense_score for the score attribute
        effective_score = self.fusion_score if self.fusion_score > 0.0 else (self.dense_score or self.lexical_score or 0.0)
        return RetrievalResult(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            user_id=self.user_id,
            chunk_index=self.chunk_index,
            chunk_text=self.chunk_text,
            page_number=self.page_number,
            filename=self.filename,
            file_type=self.file_type,
            token_count=self.token_count,
            score=effective_score,
            rank=self.rank,
        )


class HybridRetrievalResponse(BaseModel):
    """Structured response containing fused candidates and search telemetry."""
    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., description="The retrieval query string")
    results: List[RetrievalResult] = Field(default_factory=list, description="Fused candidates formatted as RetrievalResult")
    candidates: List[HybridCandidate] = Field(default_factory=list, description="Detailed candidates with dense and lexical ranks")
    total_fused: int = Field(..., ge=0, description="Total unique fused candidates after deduplication")
    dense_count: int = Field(..., ge=0, description="Number of candidates retrieved from dense vector search")
    lexical_count: int = Field(..., ge=0, description="Number of candidates retrieved from lexical search")
