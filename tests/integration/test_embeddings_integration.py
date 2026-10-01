"""Integration tests for EmbeddingService with persisted DocumentChunks and real local inference."""

import math
import pytest
from sqlalchemy.orm import Session

from backend.app.chunking.config import ChunkingConfig
from backend.app.chunking.service import ChunkingService
from backend.app.embeddings.config import EmbeddingConfig
from backend.app.embeddings.local_provider import LocalEmbeddingProvider
from backend.app.embeddings.service import EmbeddingService
from backend.app.ingestion.base import ExtractedDocument, ExtractedPage
from backend.app.models.user import User
from backend.app.models.document import Document, DocumentStatus
from backend.app.repositories.document_chunk_repository import DocumentChunkRepository


@pytest.fixture
def embedding_test_user(db_session: Session) -> User:
    user = User(
        email="embed_tester@example.com",
        hashed_password="hashed_secure_password",
        full_name="Embed Tester",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def chunked_document(db_session: Session, embedding_test_user: User) -> Document:
    doc = Document(
        user_id=embedding_test_user.id,
        filename="ai_overview.pdf",
        file_type=".pdf",
        file_size_bytes=2048,
        storage_path="storage/uploads/test/ai_overview.pdf",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    # Chunk and persist
    chunking_service = ChunkingService(config=ChunkingConfig(chunk_size=120, chunk_overlap=20))
    extracted = ExtractedDocument(
        filename="ai_overview.pdf",
        file_type=".pdf",
        pages=[
            ExtractedPage(page_number=1, text="Page 1: Dense embeddings represent semantic meaning in continuous vector space."),
            ExtractedPage(page_number=2, text="Page 2: Vector search engines like Qdrant index vectors for fast approximate nearest neighbors."),
        ],
        total_pages=2,
    )
    chunking_service.chunk_and_persist(document=doc, extracted=extracted, db=db_session)
    db_session.refresh(doc)
    return doc


class TestEmbeddingWithPersistedChunks:
    """Tests embedding generation for chunks queried from database."""

    def test_embed_persisted_document_chunks(
        self, db_session: Session, embedding_test_user: User, chunked_document: Document
    ):
        chunk_repo = DocumentChunkRepository(db_session)
        chunks = chunk_repo.get_by_document(chunked_document.id, embedding_test_user.id)

        assert len(chunks) >= 2

        # Use EmbeddingService with LocalEmbeddingProvider
        service = EmbeddingService()
        chunk_embeddings = service.embed_chunks(chunks)

        assert len(chunk_embeddings) == len(chunks)

        for chunk, embedding in zip(chunks, chunk_embeddings):
            assert embedding.chunk_id == chunk.id
            assert embedding.chunk_index == chunk.chunk_index
            assert embedding.document_id == chunk.document_id
            assert embedding.dimension == 384
            assert len(embedding.vector) == 384
            assert embedding.token_count == chunk.token_count
            assert embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"

            # Verify that DocumentChunk in database still has qdrant_point_id=None (Phase 9 boundary)
            assert chunk.qdrant_point_id is None

            # Verify all vector values are finite numeric floats
            assert all(isinstance(v, float) and not math.isnan(v) and not math.isinf(v) for v in embedding.vector)


class TestRealLocalEmbeddingInference:
    """Smoke test for real local sentence-transformers inference."""

    def test_local_model_generates_normalized_384d_vectors(self):
        config = EmbeddingConfig(
            provider="local",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            batch_size=16,
            normalize_embeddings=True,
        )
        provider = LocalEmbeddingProvider(config=config)

        sentences = [
            "Advanced Retrieval-Augmented Generation.",
            "Dense vector embeddings for neural search.",
            "Sentence transformers encode textual context.",
        ]

        vectors = provider.embed_texts(sentences)

        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 384
            # Verify L2 normalization: sum(x^2) ~= 1.0
            l2_norm = math.sqrt(sum(x * x for x in vec))
            assert math.isclose(l2_norm, 1.0, rel_tol=1e-3)
