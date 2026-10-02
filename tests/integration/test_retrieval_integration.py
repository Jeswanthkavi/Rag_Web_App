"""Integration tests for Semantic Retrieval & Similarity Search against live Qdrant and MySQL test databases."""

import pytest
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient

from backend.app.chunking.config import ChunkingConfig
from backend.app.chunking.service import ChunkingService
from backend.app.embeddings.service import EmbeddingService
from backend.app.ingestion.base import ExtractedDocument, ExtractedPage
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.user import User
from backend.app.retrieval.config import RetrievalConfig
from backend.app.retrieval.models import RetrievalResponse
from backend.app.retrieval.service import RetrievalService
from backend.app.services.document_service import DocumentService
from backend.app.vectorstore.client import create_qdrant_client
from backend.app.vectorstore.config import VectorStoreConfig
from backend.app.vectorstore.repository import QdrantVectorRepository
from backend.app.vectorstore.service import VectorStoreService
from tests.unit.test_embeddings import FakeEmbeddingProvider

TEST_RETRIEVAL_COLLECTION = "advanced_rag_test_retrieval_chunks"


@pytest.fixture(scope="module")
def qdrant_client() -> QdrantClient:
    """Provides a direct QdrantClient and ensures test collection cleanup."""
    config = VectorStoreConfig.from_settings(collection_name=TEST_RETRIEVAL_COLLECTION)
    client = create_qdrant_client(config)

    if client.collection_exists(TEST_RETRIEVAL_COLLECTION):
        client.delete_collection(TEST_RETRIEVAL_COLLECTION)

    yield client

    if client.collection_exists(TEST_RETRIEVAL_COLLECTION):
        client.delete_collection(TEST_RETRIEVAL_COLLECTION)


@pytest.fixture
def fake_embedding_service() -> EmbeddingService:
    """Deterministic 384-dimensional embedding service."""
    return EmbeddingService(provider=FakeEmbeddingProvider(dimension=384))


@pytest.fixture
def vector_repo(qdrant_client: QdrantClient) -> QdrantVectorRepository:
    config = VectorStoreConfig.from_settings(collection_name=TEST_RETRIEVAL_COLLECTION)
    return QdrantVectorRepository(client=qdrant_client, config=config)


@pytest.fixture
def vector_service(vector_repo: QdrantVectorRepository, fake_embedding_service: EmbeddingService) -> VectorStoreService:
    config = VectorStoreConfig.from_settings(collection_name=TEST_RETRIEVAL_COLLECTION)
    return VectorStoreService(
        vector_repo=vector_repo,
        embedding_service=fake_embedding_service,
        config=config,
    )


@pytest.fixture
def retrieval_service(vector_repo: QdrantVectorRepository, fake_embedding_service: EmbeddingService) -> RetrievalService:
    config = RetrievalConfig.from_settings(
        top_k=5,
        similarity_threshold=0.0,
        collection_name=TEST_RETRIEVAL_COLLECTION,
    )
    return RetrievalService(
        vector_repo=vector_repo,
        embedding_service=fake_embedding_service,
        config=config,
    )


@pytest.fixture
def tenant_user_1(db_session: Session) -> User:
    user = User(
        email="retrieval_user_1@example.com",
        hashed_password="hashed_secure_password_1",
        full_name="Tenant One",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def tenant_user_2(db_session: Session) -> User:
    user = User(
        email="retrieval_user_2@example.com",
        hashed_password="hashed_secure_password_2",
        full_name="Tenant Two",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def indexed_documents_setup(
    db_session: Session,
    vector_service: VectorStoreService,
    tenant_user_1: User,
    tenant_user_2: User,
) -> dict:
    """Creates, chunks, and indexes multiple documents across two distinct tenants."""
    chunking_service = ChunkingService(config=ChunkingConfig(chunk_size=100, chunk_overlap=20))

    # Tenant 1 - Doc A (Machine Learning overview)
    doc_1a = Document(
        user_id=tenant_user_1.id,
        filename="machine_learning.pdf",
        file_type=".pdf",
        file_size_bytes=2048,
        storage_path="storage/uploads/test/machine_learning.pdf",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc_1a)
    db_session.commit()
    db_session.refresh(doc_1a)

    ext_1a = ExtractedDocument(
        filename="machine_learning.pdf",
        file_type=".pdf",
        pages=[
            ExtractedPage(page_number=1, text="Machine learning algorithms build mathematical models based on sample data."),
            ExtractedPage(page_number=2, text="Supervised learning involves training models on labeled input-output pairs."),
        ],
        total_pages=2,
    )
    chunking_service.chunk_and_persist(document=doc_1a, extracted=ext_1a, db=db_session)
    vector_service.index_document(document=doc_1a, db=db_session, collection_name=TEST_RETRIEVAL_COLLECTION)

    # Tenant 1 - Doc B (Database Systems)
    doc_1b = Document(
        user_id=tenant_user_1.id,
        filename="databases.docx",
        file_type=".docx",
        file_size_bytes=1500,
        storage_path="storage/uploads/test/databases.docx",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc_1b)
    db_session.commit()
    db_session.refresh(doc_1b)

    ext_1b = ExtractedDocument(
        filename="databases.docx",
        file_type=".docx",
        pages=[
            ExtractedPage(page_number=1, text="Relational databases use structured query language for data manipulation."),
            ExtractedPage(page_number=2, text="ACID transactions ensure database reliability and consistency."),
        ],
        total_pages=2,
    )
    chunking_service.chunk_and_persist(document=doc_1b, extracted=ext_1b, db=db_session)
    vector_service.index_document(document=doc_1b, db=db_session, collection_name=TEST_RETRIEVAL_COLLECTION)

    # Tenant 2 - Doc C (Secret Finances)
    doc_2 = Document(
        user_id=tenant_user_2.id,
        filename="secret_finances.txt",
        file_type=".txt",
        file_size_bytes=800,
        storage_path="storage/uploads/test/secret_finances.txt",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc_2)
    db_session.commit()
    db_session.refresh(doc_2)

    ext_2 = ExtractedDocument(
        filename="secret_finances.txt",
        file_type=".txt",
        pages=[
            ExtractedPage(page_number=1, text="Confidential financial report for Q4: revenue exceeded targets by 20 percent."),
        ],
        total_pages=1,
    )
    chunking_service.chunk_and_persist(document=doc_2, extracted=ext_2, db=db_session)
    vector_service.index_document(document=doc_2, db=db_session, collection_name=TEST_RETRIEVAL_COLLECTION)

    return {
        "doc_1a": doc_1a,
        "doc_1b": doc_1b,
        "doc_2": doc_2,
        "user_1": tenant_user_1,
        "user_2": tenant_user_2,
    }


class TestSemanticRetrievalLive:
    """Integration tests executing semantic retrieval against live Qdrant and MySQL."""

    def test_basic_semantic_search_success(
        self,
        retrieval_service: RetrievalService,
        indexed_documents_setup: dict,
    ):
        user_1 = indexed_documents_setup["user_1"]
        response: RetrievalResponse = retrieval_service.search(
            query="machine learning algorithms and models",
            user_id=str(user_1.id),
            top_k=5,
            similarity_threshold=0.0,
            collection_name=TEST_RETRIEVAL_COLLECTION,
        )

        assert response.total_retrieved > 0
        assert len(response.results) == response.total_retrieved
        assert response.top_k == 5

        # Validate structure of first result
        first = response.results[0]
        assert first.rank == 1
        assert first.user_id == str(user_1.id)
        assert first.document_id in [str(indexed_documents_setup["doc_1a"].id), str(indexed_documents_setup["doc_1b"].id)]
        assert first.chunk_text is not None and len(first.chunk_text) > 0
        assert first.filename in ["machine_learning.pdf", "databases.docx"]
        assert first.file_type in [".pdf", ".docx"]
        assert 0.0 <= first.score <= 1.0

        # Validate descending score order
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_strict_tenant_isolation(
        self,
        retrieval_service: RetrievalService,
        indexed_documents_setup: dict,
    ):
        user_1 = indexed_documents_setup["user_1"]
        user_2 = indexed_documents_setup["user_2"]

        # User 1 searches for terms specifically in User 2's document
        response_1 = retrieval_service.search(
            query="Confidential financial report revenue",
            user_id=str(user_1.id),
            top_k=10,
            similarity_threshold=0.0,
            collection_name=TEST_RETRIEVAL_COLLECTION,
        )

        # None of User 1's results should belong to User 2 or contain User 2's text
        for r in response_1.results:
            assert r.user_id == str(user_1.id)
            assert r.document_id != str(indexed_documents_setup["doc_2"].id)
            assert "Confidential financial report" not in r.chunk_text

        # User 2 searches for their own document
        response_2 = retrieval_service.search(
            query="Confidential financial report revenue",
            user_id=str(user_2.id),
            top_k=5,
            similarity_threshold=0.0,
            collection_name=TEST_RETRIEVAL_COLLECTION,
        )

        assert response_2.total_retrieved >= 1
        for r in response_2.results:
            assert r.user_id == str(user_2.id)
            assert r.document_id == str(indexed_documents_setup["doc_2"].id)

    def test_top_k_limits_results(
        self,
        retrieval_service: RetrievalService,
        indexed_documents_setup: dict,
    ):
        user_1 = indexed_documents_setup["user_1"]
        response = retrieval_service.search(
            query="learning and databases",
            user_id=str(user_1.id),
            top_k=2,
            similarity_threshold=0.0,
            collection_name=TEST_RETRIEVAL_COLLECTION,
        )

        assert len(response.results) <= 2
        assert response.total_retrieved <= 2
        if len(response.results) == 2:
            assert response.results[0].rank == 1
            assert response.results[1].rank == 2

    def test_document_scoping_filter(
        self,
        retrieval_service: RetrievalService,
        indexed_documents_setup: dict,
    ):
        user_1 = indexed_documents_setup["user_1"]
        doc_1b = indexed_documents_setup["doc_1b"]

        response = retrieval_service.search(
            query="algorithms and structured data",
            user_id=str(user_1.id),
            document_id=str(doc_1b.id),
            top_k=5,
            similarity_threshold=0.0,
            collection_name=TEST_RETRIEVAL_COLLECTION,
        )

        assert response.total_retrieved > 0
        for r in response.results:
            assert r.document_id == str(doc_1b.id)
            assert r.filename == "databases.docx"

    def test_cross_tenant_document_filter_returns_empty(
        self,
        retrieval_service: RetrievalService,
        indexed_documents_setup: dict,
    ):
        user_1 = indexed_documents_setup["user_1"]
        doc_2 = indexed_documents_setup["doc_2"]  # Belongs to user 2

        # User 1 attempts to filter on User 2's document ID
        response = retrieval_service.search(
            query="Confidential financial report",
            user_id=str(user_1.id),
            document_id=str(doc_2.id),
            top_k=5,
            similarity_threshold=0.0,
            collection_name=TEST_RETRIEVAL_COLLECTION,
        )

        # Must return empty results without leaking existence
        assert response.total_retrieved == 0
        assert response.results == []

    def test_similarity_threshold_filtering(
        self,
        retrieval_service: RetrievalService,
        indexed_documents_setup: dict,
    ):
        user_1 = indexed_documents_setup["user_1"]

        # Search with an impossible threshold (0.999999)
        response = retrieval_service.search(
            query="random query string",
            user_id=str(user_1.id),
            top_k=5,
            similarity_threshold=0.999999,
            collection_name=TEST_RETRIEVAL_COLLECTION,
        )

        assert response.total_retrieved == 0
        assert response.results == []

    def test_retrieval_after_document_deletion(
        self,
        db_session: Session,
        retrieval_service: RetrievalService,
        vector_service: VectorStoreService,
        indexed_documents_setup: dict,
        tmp_path,
    ):
        user_1 = indexed_documents_setup["user_1"]
        doc_1a = indexed_documents_setup["doc_1a"]

        # Verify doc_1a was retrievable
        res_before = retrieval_service.search(
            query="Machine learning algorithms",
            user_id=str(user_1.id),
            document_id=str(doc_1a.id),
            top_k=5,
            similarity_threshold=0.0,
            collection_name=TEST_RETRIEVAL_COLLECTION,
        )
        assert res_before.total_retrieved > 0

        # Delete doc_1a via DocumentService with vector cleanup
        from backend.app.repositories.document_repository import DocumentRepository
        doc_service = DocumentService(
            doc_repo=DocumentRepository(db_session),
            vector_service=vector_service,
        )
        doc_service.delete_document(user=user_1, document_id=str(doc_1a.id))

        # Search again for doc_1a
        res_after = retrieval_service.search(
            query="Machine learning algorithms",
            user_id=str(user_1.id),
            document_id=str(doc_1a.id),
            top_k=5,
            similarity_threshold=0.0,
            collection_name=TEST_RETRIEVAL_COLLECTION,
        )
        assert res_after.total_retrieved == 0
        assert res_after.results == []
