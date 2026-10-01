"""Integration tests for ChunkingService, persistence, idempotency, and tenant isolation."""

import pytest
from unittest.mock import patch
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.chunking.config import ChunkingConfig
from backend.app.chunking.service import ChunkingService
from backend.app.ingestion.base import ExtractedDocument, ExtractedPage
from backend.app.ingestion.service import IngestionService
from backend.app.models.user import User
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_chunk import DocumentChunk
from backend.app.repositories.document_chunk_repository import DocumentChunkRepository


@pytest.fixture
def test_user(db_session: Session) -> User:
    user = User(
        email="chunking_test_user@example.com",
        hashed_password="hashed_secret_pw",
        full_name="Chunking Tester",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def other_user(db_session: Session) -> User:
    user = User(
        email="chunking_other_user@example.com",
        hashed_password="hashed_secret_pw",
        full_name="Other User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def sample_document(db_session: Session, test_user: User) -> Document:
    doc = Document(
        user_id=test_user.id,
        filename="test_guide.pdf",
        file_type=".pdf",
        file_size_bytes=1024,
        storage_path="storage/uploads/test/test_guide.pdf",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


class TestChunkingPersistenceAndMetadata:
    """Tests chunking persistence, metadata retention, and database constraints."""

    def test_chunking_persists_metadata_and_null_qdrant_point(
        self, db_session: Session, sample_document: Document, test_user: User
    ):
        service = ChunkingService(config=ChunkingConfig(chunk_size=100, chunk_overlap=20))
        extracted = ExtractedDocument(
            filename="test_guide.pdf",
            file_type=".pdf",
            pages=[
                ExtractedPage(page_number=1, text="Page 1: Deep learning representations of text."),
                ExtractedPage(page_number=2, text="Page 2: Vector indexing with hierarchical graph algorithms."),
            ],
            total_pages=2,
        )

        persisted_chunks = service.chunk_and_persist(
            document=sample_document,
            extracted=extracted,
            db=db_session,
        )

        assert len(persisted_chunks) == 2
        assert sample_document.total_chunks == 2

        # Verify from database directly
        repo = DocumentChunkRepository(db_session)
        db_chunks = repo.get_by_document(sample_document.id, test_user.id)
        assert len(db_chunks) == 2

        # Verify chunk 0
        c0 = db_chunks[0]
        assert c0.chunk_index == 0
        assert c0.document_id == sample_document.id
        assert c0.user_id == test_user.id
        assert c0.page_number == 1
        assert "Page 1" in c0.chunk_text
        assert c0.token_count > 0
        assert c0.qdrant_point_id is None  # Must remain None in Phase 8

        # Verify chunk 1
        c1 = db_chunks[1]
        assert c1.chunk_index == 1
        assert c1.document_id == sample_document.id
        assert c1.user_id == test_user.id
        assert c1.page_number == 2
        assert "Page 2" in c1.chunk_text
        assert c1.token_count > 0
        assert c1.qdrant_point_id is None

    def test_empty_and_whitespace_document_handling(
        self, db_session: Session, sample_document: Document, test_user: User
    ):
        service = ChunkingService()
        empty_extracted = ExtractedDocument(
            filename="empty.txt",
            file_type=".txt",
            pages=[ExtractedPage(page_number=None, text="   \n\t  ")],
            total_pages=1,
        )

        chunks = service.chunk_and_persist(
            document=sample_document,
            extracted=empty_extracted,
            db=db_session,
        )

        assert len(chunks) == 0
        assert sample_document.total_chunks == 0

        repo = DocumentChunkRepository(db_session)
        db_chunks = repo.get_by_document(sample_document.id, test_user.id)
        assert len(db_chunks) == 0

    def test_short_document_creates_single_chunk(
        self, db_session: Session, sample_document: Document, test_user: User
    ):
        service = ChunkingService(config=ChunkingConfig(chunk_size=500, chunk_overlap=50))
        short_extracted = ExtractedDocument(
            filename="short.txt",
            file_type=".txt",
            pages=[ExtractedPage(page_number=None, text="This is short content.")],
            total_pages=1,
        )

        chunks = service.chunk_and_persist(
            document=sample_document,
            extracted=short_extracted,
            db=db_session,
        )

        assert len(chunks) == 1
        assert sample_document.total_chunks == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].chunk_text == "This is short content."
        assert chunks[0].page_number is None

    def test_idempotency_on_repeated_runs(
        self, db_session: Session, sample_document: Document, test_user: User
    ):
        service = ChunkingService(config=ChunkingConfig(chunk_size=80, chunk_overlap=15))
        extracted = ExtractedDocument(
            filename="test_guide.pdf",
            file_type=".pdf",
            pages=[
                ExtractedPage(page_number=1, text="First sentence on page one. Second sentence on page one."),
                ExtractedPage(page_number=2, text="Third sentence on page two. Fourth sentence on page two."),
            ],
            total_pages=2,
        )

        # Run 1
        run_1_chunks = service.chunk_and_persist(
            document=sample_document,
            extracted=extracted,
            db=db_session,
        )
        run_1_count = len(run_1_chunks)
        assert run_1_count >= 2

        # Run 2 (reprocessing the exact same document)
        run_2_chunks = service.chunk_and_persist(
            document=sample_document,
            extracted=extracted,
            db=db_session,
        )
        assert len(run_2_chunks) == run_1_count
        assert sample_document.total_chunks == run_1_count

        # Verify DB records are replaced and not duplicated
        repo = DocumentChunkRepository(db_session)
        total_in_db = repo.get_by_document(sample_document.id, test_user.id)
        assert len(total_in_db) == run_1_count

    def test_multi_tenant_isolation(
        self, db_session: Session, sample_document: Document, test_user: User, other_user: User
    ):
        service = ChunkingService()
        extracted = ExtractedDocument(
            filename="user1_doc.txt",
            file_type=".txt",
            pages=[ExtractedPage(page_number=None, text="Confidential User 1 Content")],
            total_pages=1,
        )

        service.chunk_and_persist(
            document=sample_document,
            extracted=extracted,
            db=db_session,
        )

        repo = DocumentChunkRepository(db_session)

        # User 1 can see chunks
        user1_chunks = repo.get_by_document(sample_document.id, test_user.id)
        assert len(user1_chunks) == 1

        # User 2 querying User 1's document_id receives empty list
        user2_chunks = repo.get_by_document(sample_document.id, other_user.id)
        assert len(user2_chunks) == 0

    def test_transactional_rollback_on_failure(
        self, db_session: Session, sample_document: Document, test_user: User
    ):
        service = ChunkingService()
        extracted = ExtractedDocument(
            filename="fail.txt",
            file_type=".txt",
            pages=[ExtractedPage(page_number=None, text="Rollback candidate content")],
            total_pages=1,
        )

        # Simulate database commit error during chunking
        with patch.object(db_session, "commit", side_effect=RuntimeError("Simulated DB Disk Failure")):
            with pytest.raises(RuntimeError, match="Simulated DB Disk Failure"):
                service.chunk_and_persist(
                    document=sample_document,
                    extracted=extracted,
                    db=db_session,
                )

        # Verify no chunks were committed
        repo = DocumentChunkRepository(db_session)
        chunks = repo.get_by_document(sample_document.id, test_user.id)
        assert len(chunks) == 0


class TestIngestionAndChunkingPipelineIntegration:
    """Verifies complete ingestion pipeline produces persistent chunks and updates total_chunks."""

    def test_ingestion_pipeline_runs_chunking_end_to_end(
        self, db_session: Session, test_user: User, tmp_path
    ):
        # Create a sample text file
        file_path = tmp_path / "research_notes.txt"
        file_path.write_text(
            "Introduction to Advanced RAG Architectures.\n\n"
            "Chapter 1: Query Transformation and HyDE.\n\n"
            "Chapter 2: Dense and Sparse Hybrid Retrieval.\n\n"
            "Chapter 3: Cross-Encoder Reranking and Context Fusion.\n",
            encoding="utf-8",
        )

        # Create Document record
        doc = Document(
            user_id=test_user.id,
            filename="research_notes.txt",
            file_type=".txt",
            file_size_bytes=len(file_path.read_bytes()),
            storage_path=str(file_path),
            status=DocumentStatus.PENDING,
        )
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)

        # Run IngestionService
        ingestion_service = IngestionService()
        processed_doc = ingestion_service.process_document(
            document_id=doc.id,
            user_id=test_user.id,
            db=db_session,
        )

        assert processed_doc is not None
        assert processed_doc.status == DocumentStatus.COMPLETED
        assert processed_doc.page_count == 1
        assert processed_doc.total_chunks > 0

        # Check chunks in DB
        repo = DocumentChunkRepository(db_session)
        chunks = repo.get_by_document(doc.id, test_user.id)
        assert len(chunks) == processed_doc.total_chunks
        for chunk in chunks:
            assert chunk.user_id == test_user.id
            assert chunk.document_id == doc.id
            assert chunk.qdrant_point_id is None
            assert chunk.token_count > 0
            assert len(chunk.chunk_text) > 0
