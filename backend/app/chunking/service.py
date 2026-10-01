"""Chunking Service coordinating text splitting, metadata preservation, and persistent storage."""

import logging
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy.orm import Session

from backend.app.chunking.config import ChunkingConfig
from backend.app.chunking.splitter import RecursiveCharacterSplitter, ChunkData
from backend.app.models.document import Document
from backend.app.models.document_chunk import DocumentChunk
from backend.app.repositories.document_chunk_repository import DocumentChunkRepository

if TYPE_CHECKING:
    from backend.app.ingestion.base import ExtractedDocument

logger = logging.getLogger("app.chunking")


class ChunkingService:
    """Orchestrates deterministic chunking, metadata attachment, and transactional persistence."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig.from_settings()
        self.splitter = RecursiveCharacterSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )

    def generate_chunks(self, extracted: "ExtractedDocument") -> List[ChunkData]:
        """Generates chunk data items from normalized extracted document pages."""
        if not extracted or not extracted.pages:
            return []
        return self.splitter.split_document_pages(extracted.pages)

    def chunk_and_persist(
        self,
        document: Document,
        extracted: "ExtractedDocument",
        db: Session,
    ) -> List[DocumentChunk]:
        """Executes splitting, removes prior chunks for idempotency, and commits new chunks."""
        if not document or not document.id or not document.user_id:
            raise ValueError("Valid Document with persistent id and user_id is required for chunking.")

        chunk_repo = DocumentChunkRepository(db)
        chunk_data_items = self.generate_chunks(extracted)

        # Build ORM entities with strict metadata preservation
        orm_chunks: List[DocumentChunk] = []
        for item in chunk_data_items:
            orm_chunks.append(
                DocumentChunk(
                    document_id=document.id,
                    user_id=document.user_id,
                    chunk_index=item.chunk_index,
                    chunk_text=item.text,
                    page_number=item.page_number,
                    token_count=item.token_count,
                    qdrant_point_id=None,  # Intentionally None until Phase 10 (Qdrant storage)
                )
            )

        try:
            # 1. Delete any existing chunks for this document (idempotency)
            chunk_repo.delete_by_document(
                document_id=document.id,
                user_id=document.user_id,
                auto_commit=False,
            )

            # 2. Insert new chunks if any
            if orm_chunks:
                chunk_repo.create_many(orm_chunks, auto_commit=False)

            # 3. Update parent Document total_chunks
            document.total_chunks = len(orm_chunks)
            db.add(document)

            # 4. Commit atomic transaction
            db.commit()
            db.refresh(document)
            for chunk in orm_chunks:
                db.refresh(chunk)

            logger.info(
                f"Chunking completed: Document {document.id} produced {len(orm_chunks)} chunks."
            )
            return orm_chunks

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to persist chunks for Document {document.id}: {e}")
            raise
