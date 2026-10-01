"""DocumentChunk Repository for tenant-isolated persistence of chunk records."""

from typing import List
from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from backend.app.models.document_chunk import DocumentChunk


class DocumentChunkRepository:
    """Repository managing persistence for DocumentChunk entities with strict tenant isolation."""

    def __init__(self, db: Session):
        self.db = db

    def create_many(
        self,
        chunks: List[DocumentChunk],
        auto_commit: bool = True,
    ) -> List[DocumentChunk]:
        """Batch-inserts document chunk records within the database session."""
        if not chunks:
            return []
        self.db.add_all(chunks)
        if auto_commit:
            self.db.commit()
            for chunk in chunks:
                self.db.refresh(chunk)
        else:
            self.db.flush()
        return chunks

    def get_by_document(self, document_id: str, user_id: str) -> List[DocumentChunk]:
        """Retrieves all chunks for a document strictly enforcing ownership by user_id."""
        stmt = (
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.user_id == user_id,
            )
            .order_by(DocumentChunk.chunk_index.asc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def delete_by_document(
        self,
        document_id: str,
        user_id: str,
        auto_commit: bool = True,
    ) -> int:
        """Deletes all chunks for a specific document belonging to user_id."""
        stmt = delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.user_id == user_id,
        )
        result = self.db.execute(stmt)
        if auto_commit:
            self.db.commit()
        else:
            self.db.flush()
        return result.rowcount or 0
