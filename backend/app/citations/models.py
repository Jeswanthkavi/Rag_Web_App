"""Domain models for structured RAG citations."""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class Citation(BaseModel):
    """Structured citation referencing an authoritative retrieved document chunk."""
    model_config = ConfigDict(from_attributes=True)

    citation_id: str = Field(..., description="Unique citation reference identifier in generated answer (e.g., 'source_1')")
    document_id: str = Field(..., description="UUID of the referenced parent document")
    chunk_id: str = Field(..., description="UUID of the referenced document chunk")
    filename: str = Field(..., description="Original filename of the referenced document")
    page_number: Optional[int] = Field(default=None, description="Page number where chunk appears (for paged docs like PDF)")
    chunk_index: int = Field(..., ge=0, description="Sequential index of the chunk within the document")
