"""Document Repository for tenant-isolated database operations on Document entities."""

from typing import Optional, List, Tuple
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from backend.app.models.document import Document, DocumentStatus


class DocumentRepository:
    """Repository managing persistence for Document entities with strict user tenancy isolation."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        user_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        storage_path: str,
        status: DocumentStatus = DocumentStatus.PENDING,
        id: Optional[str] = None,
    ) -> Document:
        """Persists a new document catalog record."""
        doc_kwargs = {
            "user_id": user_id,
            "filename": filename,
            "file_type": file_type,
            "file_size_bytes": file_size_bytes,
            "storage_path": storage_path,
            "status": status,
        }
        if id:
            doc_kwargs["id"] = id
            
        doc = Document(**doc_kwargs)
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_by_id(self, document_id: str, user_id: str) -> Optional[Document]:
        """Retrieves a document strictly verifying ownership by user_id."""
        statement = select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
        return self.db.execute(statement).scalar_one_or_none()

    def list_by_user(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> Tuple[List[Document], int]:
        """Lists documents owned by user_id with pagination and total count."""
        # Total count query
        count_stmt = select(func.count(Document.id)).where(Document.user_id == user_id)
        total = self.db.execute(count_stmt).scalar() or 0

        # Items query
        items_stmt = (
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        items = list(self.db.execute(items_stmt).scalars().all())
        return items, total

    def delete(self, document: Document) -> None:
        """Deletes a document record, triggering cascading removal of any child chunks."""
        self.db.delete(document)
        self.db.commit()
