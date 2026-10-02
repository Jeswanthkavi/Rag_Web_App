"""Document Ingestion Service coordinating extraction, normalization, chunking, and status transitions."""

import logging
from pathlib import Path
from typing import Dict, Type, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.db.session import SessionLocal
from backend.app.models.document import Document, DocumentStatus
from backend.app.ingestion.base import BaseExtractor, ExtractedDocument, ExtractedPage
from backend.app.ingestion.pdf_extractor import PDFExtractor
from backend.app.ingestion.docx_extractor import DOCXExtractor
from backend.app.ingestion.txt_extractor import TXTExtractor
from backend.app.ingestion.normalizer import TextNormalizer
from backend.app.chunking.service import ChunkingService

logger = logging.getLogger("app.ingestion")


class IngestionService:
    """Orchestrates document parsing, text extraction, normalization, and chunking."""

    _EXTRACTOR_REGISTRY: Dict[str, Type[BaseExtractor]] = {
        ".pdf": PDFExtractor,
        ".docx": DOCXExtractor,
        ".txt": TXTExtractor,
    }

    def __init__(self, chunking_service: Optional[ChunkingService] = None):
        self.chunking_service = chunking_service or ChunkingService()

    def get_extractor(self, file_type: str) -> BaseExtractor:
        """Returns the appropriate extractor for the file extension."""
        ext = file_type.lower()
        extractor_cls = self._EXTRACTOR_REGISTRY.get(ext)
        if not extractor_cls:
            raise ValueError(f"No extractor registered for file type '{file_type}'")
        return extractor_cls()

    def extract_and_normalize(
        self,
        file_path: Path,
        filename: str,
        file_type: str,
    ) -> ExtractedDocument:
        """Extracts text and applies normalization across all extracted pages."""
        extractor = self.get_extractor(file_type)
        raw_doc = extractor.extract(file_path=file_path, filename=filename)

        normalized_pages = []
        for page in raw_doc.pages:
            clean_text = TextNormalizer.normalize(page.text)
            normalized_pages.append(
                ExtractedPage(
                    page_number=page.page_number,
                    text=clean_text,
                    metadata=page.metadata,
                )
            )

        return ExtractedDocument(
            filename=raw_doc.filename,
            file_type=raw_doc.file_type,
            pages=normalized_pages,
            total_pages=raw_doc.total_pages,
            metadata=raw_doc.metadata,
        )

    def process_document(
        self,
        document_id: str,
        user_id: str,
        db: Session,
    ) -> Optional[Document]:
        """Executes ingestion lifecycle (PENDING -> PROCESSING -> COMPLETED/FAILED) with DB persistence."""
        # 1. Fetch document strictly enforcing tenancy
        doc_stmt = select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
        document = db.execute(doc_stmt).scalar_one_or_none()
        if not document:
            logger.warning(f"Ingestion aborted: Document {document_id} not found for user {user_id}")
            return None

        # 2. Transition status to PROCESSING
        document.status = DocumentStatus.PROCESSING
        document.error_message = None
        db.commit()
        db.refresh(document)

        # 3. Perform Extraction & Normalization
        try:
            file_path = Path(document.storage_path)
            extracted = self.extract_and_normalize(
                file_path=file_path,
                filename=document.filename,
                file_type=document.file_type,
            )

            # 4. Perform Chunking & Chunk Persistence
            self.chunking_service.chunk_and_persist(
                document=document,
                extracted=extracted,
                db=db,
            )

            # 5. Finalize document record
            document.status = DocumentStatus.COMPLETED
            document.page_count = extracted.total_pages
            document.error_message = None
            db.commit()
            db.refresh(document)

            # Invalidate tenant BM25 cache
            from backend.app.hybrid_retrieval.cache import get_bm25_cache
            get_bm25_cache().invalidate_user(user_id=str(user_id))

            logger.info(
                f"Ingestion successful: Document {document_id} marked COMPLETED "
                f"(pages: {document.page_count}, chunks: {document.total_chunks})"
            )
            return document

        except Exception as e:
            # 6. Handle and record sanitized failure
            db.rollback()
            sanitized_error = str(e).replace("\\", "/").split(":")[-1].strip()
            if not sanitized_error:
                sanitized_error = "Extraction, normalization, or chunking failed."

            document.status = DocumentStatus.FAILED
            document.error_message = sanitized_error
            db.commit()
            db.refresh(document)
            logger.error(f"Ingestion failed for Document {document_id}: {sanitized_error}")
            return document


def run_document_ingestion(document_id: str, user_id: str, session_factory=None) -> None:
    """Background task entry point for asynchronous document ingestion."""
    factory = session_factory or SessionLocal
    db: Session = factory()
    try:
        service = IngestionService()
        service.process_document(document_id=document_id, user_id=user_id, db=db)
    finally:
        db.close()
