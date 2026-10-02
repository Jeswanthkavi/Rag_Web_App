"""Integration tests for Qdrant VectorStore operations against live Qdrant instance."""

import pytest
from unittest.mock import patch
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient

from backend.app.chunking.config import ChunkingConfig
from backend.app.chunking.service import ChunkingService
from backend.app.embeddings.service import EmbeddingService
from backend.app.ingestion.base import ExtractedDocument, ExtractedPage
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_chunk import DocumentChunk
from backend.app.models.user import User
from backend.app.repositories.document_chunk_repository import DocumentChunkRepository
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.services.document_service import DocumentService
from backend.app.vectorstore.client import create_qdrant_client
from backend.app.vectorstore.config import VectorStoreConfig
from backend.app.vectorstore.models import VectorStoreError
from backend.app.vectorstore.repository import QdrantVectorRepository
from backend.app.vectorstore.service import VectorStoreService

from tests.unit.test_embeddings import FakeEmbeddingProvider

TEST_COLLECTION = "advanced_rag_test_chunks"


@pytest.fixture(scope="module")
def qdrant_test_client() -> QdrantClient:
    """Provides a direct QdrantClient and ensures test collection cleanup."""
    config = VectorStoreConfig.from_settings(collection_name=TEST_COLLECTION)
    client = create_qdrant_client(config)

    # Cleanup before run
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)

    yield client

    # Cleanup after run
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)


@pytest.fixture
def vector_service(qdrant_test_client: QdrantClient) -> VectorStoreService:
    config = VectorStoreConfig.from_settings(collection_name=TEST_COLLECTION)
    repo = QdrantVectorRepository(client=qdrant_test_client, config=config)
    embedding_service = EmbeddingService(provider=FakeEmbeddingProvider(dimension=384))
    return VectorStoreService(vector_repo=repo, embedding_service=embedding_service, config=config)


@pytest.fixture
def user_a(db_session: Session) -> User:
    user = User(
        email="tenant_a@example.com",
        hashed_password="hashed_secure_password_a",
        full_name="Tenant A",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def user_b(db_session: Session) -> User:
    user = User(
        email="tenant_b@example.com",
        hashed_password="hashed_secure_password_b",
        full_name="Tenant B",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def chunked_doc_user_a(db_session: Session, user_a: User) -> Document:
    doc = Document(
        user_id=user_a.id,
        filename="neural_search.pdf",
        file_type=".pdf",
        file_size_bytes=1024,
        storage_path="storage/uploads/test/neural_search.pdf",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    # Chunk and persist
    chunking_service = ChunkingService(config=ChunkingConfig(chunk_size=100, chunk_overlap=20))
    extracted = ExtractedDocument(
        filename="neural_search.pdf",
        file_type=".pdf",
        pages=[
            ExtractedPage(page_number=1, text="Page 1: Approximate nearest neighbors via HNSW graphs."),
            ExtractedPage(page_number=2, text="Page 2: Cosine similarity in dense vector spaces."),
        ],
        total_pages=2,
    )
    chunking_service.chunk_and_persist(document=doc, extracted=extracted, db=db_session)
    db_session.refresh(doc)
    return doc


class TestQdrantIntegrationLive:
    """Live integration tests for collection management, upserts, tenant isolation, and deletions."""

    def test_collection_initialization_and_health(
        self, vector_service: VectorStoreService, qdrant_test_client: QdrantClient
    ):
        # 1. Initialize collection
        init_res = vector_service.initialize_collection(TEST_COLLECTION)
        assert init_res is True
        assert qdrant_test_client.collection_exists(TEST_COLLECTION) is True

        # 2. Re-initialization is idempotent
        reinit_res = vector_service.initialize_collection(TEST_COLLECTION)
        assert reinit_res is True

        # 3. Health check
        health = vector_service.check_health(TEST_COLLECTION)
        assert health.status == "healthy"
        assert health.collection_name == TEST_COLLECTION
        assert health.collection_exists is True
        assert health.dimension == 384
        assert "COSINE" in health.distance.upper()

    def test_index_document_and_mysql_sync(
        self,
        db_session: Session,
        vector_service: VectorStoreService,
        chunked_doc_user_a: Document,
        user_a: User,
        qdrant_test_client: QdrantClient,
    ):
        result = vector_service.index_document(
            document=chunked_doc_user_a,
            db=db_session,
            collection_name=TEST_COLLECTION,
        )

        assert result.indexed_chunks_count >= 2
        assert len(result.point_ids) == result.indexed_chunks_count

        # Verify points exist in Qdrant
        repo = DocumentChunkRepository(db_session)
        chunks = repo.get_by_document(chunked_doc_user_a.id, user_a.id)

        for chunk in chunks:
            # Verify MySQL qdrant_point_id is populated with deterministic ID matching chunk.id
            assert chunk.qdrant_point_id == chunk.id

            # Verify point in Qdrant
            point_data = vector_service.vector_repo.get_point(
                point_id=chunk.id,
                collection_name=TEST_COLLECTION,
            )
            assert point_data is not None
            assert point_data["id"] == chunk.id
            assert len(point_data["vector"]) == 384
            assert point_data["payload"]["document_id"] == chunked_doc_user_a.id
            assert point_data["payload"]["user_id"] == user_a.id
            assert point_data["payload"]["chunk_text"] == chunk.chunk_text

    def test_reindexing_idempotency(
        self,
        db_session: Session,
        vector_service: VectorStoreService,
        chunked_doc_user_a: Document,
        user_a: User,
    ):
        # Run 1
        res1 = vector_service.index_document(
            document=chunked_doc_user_a,
            db=db_session,
            collection_name=TEST_COLLECTION,
        )

        # Run 2 (reindexing the same document)
        res2 = vector_service.index_document(
            document=chunked_doc_user_a,
            db=db_session,
            collection_name=TEST_COLLECTION,
        )

        assert res1.indexed_chunks_count == res2.indexed_chunks_count
        assert res1.point_ids == res2.point_ids

        # Total points in collection for user_a should equal exact chunk count (no duplicates)
        total_points = vector_service.vector_repo.count_points(
            collection_name=TEST_COLLECTION,
            user_id=user_a.id,
            document_id=chunked_doc_user_a.id,
        )
        assert total_points == res1.indexed_chunks_count

    def test_reindex_with_modified_chunks_purges_stale_points(
        self,
        db_session: Session,
        vector_service: VectorStoreService,
        chunked_doc_user_a: Document,
        user_a: User,
    ):
        # 1. First indexing of initial chunks
        res1 = vector_service.index_document(
            document=chunked_doc_user_a,
            db=db_session,
            collection_name=TEST_COLLECTION,
        )
        old_point_ids = set(res1.point_ids)
        assert len(old_point_ids) >= 2

        # 2. Simulate document re-chunking in Phase 8 (new chunk UUIDs created)
        chunk_repo = DocumentChunkRepository(db_session)
        chunk_repo.delete_by_document(chunked_doc_user_a.id, user_a.id)

        new_chunks = [
            DocumentChunk(
                document_id=chunked_doc_user_a.id,
                user_id=user_a.id,
                chunk_index=0,
                chunk_text="Updated content version 2 chunk A.",
                token_count=6,
            ),
            DocumentChunk(
                document_id=chunked_doc_user_a.id,
                user_id=user_a.id,
                chunk_index=1,
                chunk_text="Updated content version 2 chunk B.",
                token_count=6,
            ),
            DocumentChunk(
                document_id=chunked_doc_user_a.id,
                user_id=user_a.id,
                chunk_index=2,
                chunk_text="Updated content version 2 chunk C.",
                token_count=6,
            ),
        ]
        chunk_repo.create_many(new_chunks)
        new_point_ids = {c.id for c in new_chunks}

        # Verify old point IDs are different from new point IDs
        assert old_point_ids.isdisjoint(new_point_ids)

        # 3. Reindex document
        res2 = vector_service.index_document(
            document=chunked_doc_user_a,
            db=db_session,
            collection_name=TEST_COLLECTION,
        )

        assert res2.indexed_chunks_count == 3
        assert set(res2.point_ids) == new_point_ids

        # 4. Verify all new points exist in Qdrant
        for p_id in new_point_ids:
            pt = vector_service.vector_repo.get_point(p_id, collection_name=TEST_COLLECTION)
            assert pt is not None

        # 5. Verify all old stale points were deleted from Qdrant
        for old_id in old_point_ids:
            pt = vector_service.vector_repo.get_point(old_id, collection_name=TEST_COLLECTION)
            assert pt is None

        # 6. Verify total points count in Qdrant for this document is exactly 3
        total_points = vector_service.vector_repo.count_points(
            collection_name=TEST_COLLECTION,
            user_id=user_a.id,
            document_id=chunked_doc_user_a.id,
        )
        assert total_points == 3

    def test_failed_upsert_preserves_existing_qdrant_vectors(
        self,
        db_session: Session,
        vector_service: VectorStoreService,
        chunked_doc_user_a: Document,
        user_a: User,
    ):
        # 1. Index document initially with valid vectors
        res1 = vector_service.index_document(
            document=chunked_doc_user_a,
            db=db_session,
            collection_name=TEST_COLLECTION,
        )
        initial_point_ids = list(res1.point_ids)
        assert len(initial_point_ids) >= 2

        # 2. Simulate an upsert failure during a subsequent reindex attempt
        with patch.object(vector_service.vector_repo, "upsert_points", side_effect=RuntimeError("Simulated Network Error")):
            with pytest.raises(VectorStoreError, match="Document indexing failed"):
                vector_service.index_document(
                    document=chunked_doc_user_a,
                    db=db_session,
                    collection_name=TEST_COLLECTION,
                )

        # 3. Verify that the previous valid points are STILL intact in Qdrant
        for p_id in initial_point_ids:
            pt = vector_service.vector_repo.get_point(p_id, collection_name=TEST_COLLECTION)
            assert pt is not None
            assert pt["id"] == p_id
            assert len(pt["vector"]) == 384

    def test_tenant_isolation_in_vectorstore(
        self,
        db_session: Session,
        vector_service: VectorStoreService,
        chunked_doc_user_a: Document,
        user_a: User,
        user_b: User,
    ):
        vector_service.index_document(
            document=chunked_doc_user_a,
            db=db_session,
            collection_name=TEST_COLLECTION,
        )

        # Count scoped to User A
        count_user_a = vector_service.vector_repo.count_points(
            collection_name=TEST_COLLECTION,
            user_id=user_a.id,
        )
        assert count_user_a >= 2

        # Count scoped to User B
        count_user_b = vector_service.vector_repo.count_points(
            collection_name=TEST_COLLECTION,
            user_id=user_b.id,
        )
        assert count_user_b == 0

    def test_document_deletion_synchronization(
        self,
        db_session: Session,
        vector_service: VectorStoreService,
        chunked_doc_user_a: Document,
        user_a: User,
        tmp_path,
    ):
        # Index document first
        vector_service.index_document(
            document=chunked_doc_user_a,
            db=db_session,
            collection_name=TEST_COLLECTION,
        )

        doc_count_before = vector_service.vector_repo.count_points(
            collection_name=TEST_COLLECTION,
            user_id=user_a.id,
            document_id=chunked_doc_user_a.id,
        )
        assert doc_count_before >= 2

        # Delete document via DocumentService
        doc_repo = DocumentRepository(db_session)
        doc_service = DocumentService(doc_repo=doc_repo, vector_service=vector_service)
        doc_service.delete_document(user=user_a, document_id=chunked_doc_user_a.id)

        # Verify Qdrant points are purged
        doc_count_after = vector_service.vector_repo.count_points(
            collection_name=TEST_COLLECTION,
            user_id=user_a.id,
            document_id=chunked_doc_user_a.id,
        )
        assert doc_count_after == 0
