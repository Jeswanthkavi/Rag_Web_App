"""Integration tests for Grounded RAG Generation against live Qdrant and MySQL test databases."""

import pytest
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient

from backend.app.chunking.config import ChunkingConfig
from backend.app.chunking.service import ChunkingService
from backend.app.embeddings.service import EmbeddingService
from backend.app.ingestion.base import ExtractedDocument, ExtractedPage
from backend.app.llm.base import LLMProviderError
from backend.app.llm.fake_provider import FakeLLMProvider
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.user import User
from backend.app.rag.config import RAGConfig, DEFAULT_NO_CONTEXT_MESSAGE
from backend.app.rag.models import RAGError, RAGResponse
from backend.app.rag.service import RAGService
from backend.app.retrieval.config import RetrievalConfig
from backend.app.retrieval.service import RetrievalService
from backend.app.services.document_service import DocumentService
from backend.app.vectorstore.client import create_qdrant_client
from backend.app.vectorstore.config import VectorStoreConfig
from backend.app.vectorstore.repository import QdrantVectorRepository
from backend.app.vectorstore.service import VectorStoreService
from tests.unit.test_embeddings import FakeEmbeddingProvider

TEST_RAG_COLLECTION = "advanced_rag_test_rag_generation_chunks"


@pytest.fixture(scope="module")
def qdrant_client() -> QdrantClient:
    """Provides a direct QdrantClient and ensures test collection cleanup."""
    config = VectorStoreConfig.from_settings(collection_name=TEST_RAG_COLLECTION)
    client = create_qdrant_client(config)

    if client.collection_exists(TEST_RAG_COLLECTION):
        client.delete_collection(TEST_RAG_COLLECTION)

    yield client

    if client.collection_exists(TEST_RAG_COLLECTION):
        client.delete_collection(TEST_RAG_COLLECTION)


@pytest.fixture
def fake_embedding_service() -> EmbeddingService:
    """Deterministic 384-dimensional embedding service."""
    return EmbeddingService(provider=FakeEmbeddingProvider(dimension=384))


@pytest.fixture
def vector_repo(qdrant_client: QdrantClient) -> QdrantVectorRepository:
    config = VectorStoreConfig.from_settings(collection_name=TEST_RAG_COLLECTION)
    return QdrantVectorRepository(client=qdrant_client, config=config)


@pytest.fixture
def vector_service(vector_repo: QdrantVectorRepository, fake_embedding_service: EmbeddingService) -> VectorStoreService:
    config = VectorStoreConfig.from_settings(collection_name=TEST_RAG_COLLECTION)
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
        collection_name=TEST_RAG_COLLECTION,
    )
    return RetrievalService(
        vector_repo=vector_repo,
        embedding_service=fake_embedding_service,
        config=config,
    )


@pytest.fixture
def tenant_user_1(db_session: Session) -> User:
    user = User(
        email="rag_tenant_1@example.com",
        hashed_password="hashed_secure_password_1",
        full_name="RAG Tenant One",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def tenant_user_2(db_session: Session) -> User:
    user = User(
        email="rag_tenant_2@example.com",
        hashed_password="hashed_secure_password_2",
        full_name="RAG Tenant Two",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def indexed_rag_documents(
    db_session: Session,
    vector_service: VectorStoreService,
    tenant_user_1: User,
    tenant_user_2: User,
) -> dict:
    """Creates, chunks, and indexes documents for multi-tenant RAG tests."""
    chunking_service = ChunkingService(config=ChunkingConfig(chunk_size=100, chunk_overlap=20))

    # Tenant 1 - Cloud Architecture Document
    doc_1a = Document(
        user_id=tenant_user_1.id,
        filename="cloud_architecture.pdf",
        file_type=".pdf",
        file_size_bytes=2048,
        storage_path="storage/uploads/test/cloud_architecture.pdf",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc_1a)
    db_session.commit()
    db_session.refresh(doc_1a)

    ext_1a = ExtractedDocument(
        filename="cloud_architecture.pdf",
        file_type=".pdf",
        pages=[
            ExtractedPage(page_number=1, text="The system uses microservices deployed on Kubernetes with auto-scaling."),
            ExtractedPage(page_number=2, text="Database replication provides a recovery point objective of 5 minutes."),
        ],
        total_pages=2,
    )
    chunking_service.chunk_and_persist(document=doc_1a, extracted=ext_1a, db=db_session)
    vector_service.index_document(document=doc_1a, db=db_session, collection_name=TEST_RAG_COLLECTION)

    # Tenant 1 - Security Policy
    doc_1b = Document(
        user_id=tenant_user_1.id,
        filename="security_policy.docx",
        file_type=".docx",
        file_size_bytes=1500,
        storage_path="storage/uploads/test/security_policy.docx",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc_1b)
    db_session.commit()
    db_session.refresh(doc_1b)

    ext_1b = ExtractedDocument(
        filename="security_policy.docx",
        file_type=".docx",
        pages=[
            ExtractedPage(page_number=1, text="All corporate passwords must be rotated every 90 days and use MFA."),
        ],
        total_pages=1,
    )
    chunking_service.chunk_and_persist(document=doc_1b, extracted=ext_1b, db=db_session)
    vector_service.index_document(document=doc_1b, db=db_session, collection_name=TEST_RAG_COLLECTION)

    # Tenant 2 - Private Financial Statement
    doc_2 = Document(
        user_id=tenant_user_2.id,
        filename="private_ledger.txt",
        file_type=".txt",
        file_size_bytes=800,
        storage_path="storage/uploads/test/private_ledger.txt",
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc_2)
    db_session.commit()
    db_session.refresh(doc_2)

    ext_2 = ExtractedDocument(
        filename="private_ledger.txt",
        file_type=".txt",
        pages=[
            ExtractedPage(page_number=1, text="Confidential: Tenant Two net revenue was 15 million in FY2025."),
        ],
        total_pages=1,
    )
    chunking_service.chunk_and_persist(document=doc_2, extracted=ext_2, db=db_session)
    vector_service.index_document(document=doc_2, db=db_session, collection_name=TEST_RAG_COLLECTION)

    return {
        "doc_1a": doc_1a,
        "doc_1b": doc_1b,
        "doc_2": doc_2,
        "user_1": tenant_user_1,
        "user_2": tenant_user_2,
    }


class TestGroundedRAGIntegration:
    """End-to-end integration tests for RAGService with live vector store and DB."""

    def test_e2e_rag_generation_success(
        self,
        retrieval_service: RetrievalService,
        indexed_rag_documents: dict,
    ):
        user_1 = indexed_rag_documents["user_1"]
        fake_llm = FakeLLMProvider(
            model_name="gemini-2.5-flash",
            custom_response="Based on the architecture document, the system deploys microservices on Kubernetes with auto-scaling.",
        )
        rag_service = RAGService(
            retrieval_service=retrieval_service,
            llm_provider=fake_llm,
        )

        resp: RAGResponse = rag_service.generate_answer(
            query="How is the system deployed and scaled?",
            user_id=str(user_1.id),
            top_k=3,
            similarity_threshold=0.0,
        )

        assert resp.query == "How is the system deployed and scaled?"
        assert "Kubernetes with auto-scaling" in resp.answer
        assert resp.retrieved_context_count > 0
        assert resp.model_name == "gemini-2.5-flash"

        # Verify LLM received structured context and system instruction
        assert fake_llm.last_prompt is not None
        assert "=== RETRIEVED CONTEXT ===" in fake_llm.last_prompt
        assert "cloud_architecture.pdf" in fake_llm.last_prompt
        assert "microservices deployed on Kubernetes" in fake_llm.last_prompt
        assert fake_llm.last_system_instruction is not None
        assert "ONLY the factual information present in the RETRIEVED CONTEXT" in fake_llm.last_system_instruction

    def test_strict_tenant_isolation_in_generation(
        self,
        retrieval_service: RetrievalService,
        indexed_rag_documents: dict,
    ):
        user_1 = indexed_rag_documents["user_1"]
        user_2 = indexed_rag_documents["user_2"]

        fake_llm_1 = FakeLLMProvider(model_name="gemini-2.5-flash")
        rag_service_1 = RAGService(retrieval_service=retrieval_service, llm_provider=fake_llm_1)

        # User 1 queries about User 2's secret finances
        rag_service_1.generate_answer(
            query="What was the net revenue in FY2025?",
            user_id=str(user_1.id),
            top_k=5,
            similarity_threshold=0.0,
        )

        # Ensure User 1's context prompt NEVER contained User 2's data
        if fake_llm_1.last_prompt:
            assert "private_ledger.txt" not in fake_llm_1.last_prompt
            assert "Tenant Two net revenue" not in fake_llm_1.last_prompt

        # User 2 queries their own document
        fake_llm_2 = FakeLLMProvider(
            model_name="gemini-2.5-flash",
            custom_response="Net revenue was 15 million in FY2025.",
        )
        rag_service_2 = RAGService(retrieval_service=retrieval_service, llm_provider=fake_llm_2)

        resp_2 = rag_service_2.generate_answer(
            query="What was the net revenue in FY2025?",
            user_id=str(user_2.id),
            top_k=5,
            similarity_threshold=0.0,
        )

        assert resp_2.retrieved_context_count >= 1
        assert "private_ledger.txt" in fake_llm_2.last_prompt
        assert "15 million in FY2025" in resp_2.answer

    def test_document_scoping_filter_in_generation(
        self,
        retrieval_service: RetrievalService,
        indexed_rag_documents: dict,
    ):
        user_1 = indexed_rag_documents["user_1"]
        doc_1b = indexed_rag_documents["doc_1b"]  # Security policy

        fake_llm = FakeLLMProvider(
            model_name="gemini-2.5-flash",
            custom_response="Corporate passwords must be rotated every 90 days and require MFA.",
        )
        rag_service = RAGService(
            retrieval_service=retrieval_service,
            llm_provider=fake_llm,
        )

        resp = rag_service.generate_answer(
            query="What are the rotation policies?",
            user_id=str(user_1.id),
            document_id=str(doc_1b.id),
            top_k=5,
            similarity_threshold=0.0,
        )

        assert resp.retrieved_context_count >= 1
        assert "security_policy.docx" in fake_llm.last_prompt
        assert "cloud_architecture.pdf" not in fake_llm.last_prompt

    def test_no_context_query_does_not_call_llm(
        self,
        retrieval_service: RetrievalService,
        indexed_rag_documents: dict,
    ):
        user_1 = indexed_rag_documents["user_1"]

        fake_llm = FakeLLMProvider(model_name="gemini-2.5-flash")
        rag_service = RAGService(
            retrieval_service=retrieval_service,
            llm_provider=fake_llm,
        )

        # Search with non-existent document_id so 0 chunks are returned
        resp = rag_service.generate_answer(
            query="completely unrelated query",
            user_id=str(user_1.id),
            document_id="00000000-0000-0000-0000-000000000000",
        )

        assert resp.retrieved_context_count == 0
        assert resp.answer == DEFAULT_NO_CONTEXT_MESSAGE
        assert fake_llm.call_count == 0
        assert fake_llm.last_prompt is None

    def test_llm_provider_failure_produces_controlled_error(
        self,
        retrieval_service: RetrievalService,
        indexed_rag_documents: dict,
    ):
        user_1 = indexed_rag_documents["user_1"]

        class FailingLLMProvider(FakeLLMProvider):
            def generate(self, prompt: str, **kwargs):
                raise LLMProviderError("Upstream Gemini API returned 503 Service Unavailable")

        rag_service = RAGService(
            retrieval_service=retrieval_service,
            llm_provider=FailingLLMProvider(),
        )

        with pytest.raises(RAGError, match="Answer generation failed"):
            rag_service.generate_answer(
                query="How is the system deployed?",
                user_id=str(user_1.id),
                similarity_threshold=0.0,
            )

    def test_deleted_document_not_used_in_rag_generation(
        self,
        db_session: Session,
        retrieval_service: RetrievalService,
        vector_service: VectorStoreService,
        indexed_rag_documents: dict,
    ):
        user_1 = indexed_rag_documents["user_1"]
        doc_1b = indexed_rag_documents["doc_1b"]

        fake_llm = FakeLLMProvider(model_name="gemini-2.5-flash")
        rag_service = RAGService(
            retrieval_service=retrieval_service,
            llm_provider=fake_llm,
        )

        # Delete security_policy doc
        from backend.app.repositories.document_repository import DocumentRepository
        doc_service = DocumentService(
            doc_repo=DocumentRepository(db_session),
            vector_service=vector_service,
        )
        doc_service.delete_document(user=user_1, document_id=str(doc_1b.id))

        # Query specifically for doc_1b
        resp = rag_service.generate_answer(
            query="What are password rotation policies?",
            user_id=str(user_1.id),
            document_id=str(doc_1b.id),
            similarity_threshold=0.0,
        )

        assert resp.retrieved_context_count == 0
        assert resp.answer == DEFAULT_NO_CONTEXT_MESSAGE
        assert fake_llm.call_count == 0

    def test_e2e_rag_generation_with_citations(
        self,
        retrieval_service: RetrievalService,
        indexed_rag_documents: dict,
    ):
        user_1 = indexed_rag_documents["user_1"]
        doc_1a = indexed_rag_documents["doc_1a"]
        fake_llm = FakeLLMProvider(
            model_name="gemini-2.5-flash",
            custom_response="The architecture uses microservices with Kubernetes [Source 1].",
        )
        rag_service = RAGService(
            retrieval_service=retrieval_service,
            llm_provider=fake_llm,
        )

        resp = rag_service.generate_answer(
            query="How is the system deployed?",
            user_id=str(user_1.id),
            document_id=str(doc_1a.id),
            top_k=2,
            similarity_threshold=0.0,
        )

        assert len(resp.citations) >= 1
        cit = resp.citations[0]
        assert cit.citation_id == "source_1"
        assert cit.document_id == str(doc_1a.id)
        assert cit.filename == "cloud_architecture.pdf"
        assert cit.page_number is not None

    def test_multi_turn_conversation_flow(
        self,
        db_session: Session,
        retrieval_service: RetrievalService,
        indexed_rag_documents: dict,
    ):
        user_1 = indexed_rag_documents["user_1"]
        from backend.app.services.conversation_service import ConversationService
        conv_service = ConversationService(db=db_session)

        # 1. Create a conversation
        conv = conv_service.create_conversation(user_id=str(user_1.id), title="DevOps Chat")

        fake_llm = FakeLLMProvider(
            model_name="gemini-2.5-flash",
            custom_response="The system uses auto-scaling on Kubernetes [Source 1].",
        )
        rag_service = RAGService(
            retrieval_service=retrieval_service,
            llm_provider=fake_llm,
            conversation_service=conv_service,
        )

        # 2. Turn 1: Ask initial question
        resp_1 = rag_service.generate_answer(
            query="How is the application deployed?",
            user_id=str(user_1.id),
            conversation_id=conv.id,
            similarity_threshold=0.0,
        )

        assert resp_1.conversation_id == conv.id
        assert resp_1.message_id is not None
        assert len(resp_1.citations) >= 1

        # 3. Turn 2: Follow-up question
        fake_llm.last_prompt = None
        fake_llm._custom_response = "Database replication has an RPO of 5 minutes [Source 1]."

        resp_2 = rag_service.generate_answer(
            query="What is the database recovery objective?",
            user_id=str(user_1.id),
            conversation_id=conv.id,
            similarity_threshold=0.0,
        )

        # Verify prompt received prior conversation history
        assert fake_llm.last_prompt is not None
        assert "=== CONVERSATION HISTORY" in fake_llm.last_prompt
        assert "User: How is the application deployed?" in fake_llm.last_prompt
        assert "The system uses auto-scaling on Kubernetes" in fake_llm.last_prompt

        # Verify message persistence in DB
        messages = conv_service.msg_repo.list_by_conversation(conv.id)
        assert len(messages) == 4
        assert messages[0].content == "How is the application deployed?"
        assert messages[1].citations is not None
        assert messages[2].content == "What is the database recovery objective?"
        assert messages[3].citations is not None

    def test_cross_tenant_conversation_rejected_in_rag(
        self,
        db_session: Session,
        retrieval_service: RetrievalService,
        indexed_rag_documents: dict,
    ):
        user_1 = indexed_rag_documents["user_1"]
        user_2 = indexed_rag_documents["user_2"]
        from backend.app.services.conversation_service import ConversationService
        conv_service = ConversationService(db=db_session)

        # User 1 creates conversation
        conv_1 = conv_service.create_conversation(user_id=str(user_1.id), title="User 1 Thread")

        rag_service = RAGService(
            retrieval_service=retrieval_service,
            llm_provider=FakeLLMProvider(),
            conversation_service=conv_service,
        )

        # User 2 attempts to query User 1's conversation
        from backend.app.rag.models import RAGValidationError
        with pytest.raises(RAGValidationError, match="not found or unauthorized"):
            rag_service.generate_answer(
                query="Unauthorized query",
                user_id=str(user_2.id),
                conversation_id=conv_1.id,
            )

