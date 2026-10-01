"""Document Pydantic schemas."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from backend.app.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """Public metadata response for an uploaded document.
    
    Excludes server filesystem storage paths and internal credentials.
    """
    id: str = Field(description="Unique document UUID")
    filename: str = Field(description="Original sanitized filename")
    file_type: str = Field(description="File extension type (.pdf, .docx, .txt)")
    file_size_bytes: int = Field(description="File size in bytes")
    status: DocumentStatus = Field(description="Document ingestion lifecycle status")
    error_message: Optional[str] = Field(default=None, description="Processing error message if failed")
    page_count: Optional[int] = Field(default=None, description="Total pages extracted if applicable")
    total_chunks: int = Field(default=0, description="Total chunks generated during ingestion")
    created_at: datetime = Field(description="Upload creation timestamp (UTC)")
    updated_at: datetime = Field(description="Last update timestamp (UTC)")

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Paginated list of document metadata items owned by current user."""
    items: List[DocumentResponse] = Field(description="List of document records")
    total: int = Field(description="Total count of documents owned by user")
    skip: int = Field(default=0, description="Pagination offset")
    limit: int = Field(default=100, description="Pagination limit")
