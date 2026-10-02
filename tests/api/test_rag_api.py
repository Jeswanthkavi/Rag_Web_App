"""API endpoint tests for POST /api/v1/rag/query."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.main import app
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.rag.models import RAGError, RAGResponse, RAGValidationError
from backend.app.rag.service import RAGService
from backend.app.api.dependencies import get_rag_service, get_current_active_user


@pytest.fixture
def client(db_session: Session) -> TestClient:
    """Provides a TestClient with database session overridden to test database."""
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def test_user(db_session: Session) -> User:
    user = User(
        email="rag_api_user@example.com",
        hashed_password="hashed_secure_password",
        full_name="RAG Tester",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user: User) -> dict:
    from backend.app.core.security import create_access_token
    token = create_access_token(subject=str(test_user.id), email=test_user.email)
    return {"Authorization": f"Bearer {token}"}


class TestRAGAPI:
    """Tests for POST /api/v1/rag/query."""

    def test_unauthenticated_query_rejected(self, client: TestClient):
        response = client.post(
            "/api/v1/rag/query",
            json={"query": "What is grounded generation?"},
        )
        assert response.status_code == 401

    def test_empty_query_rejected(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/api/v1/rag/query",
            json={"query": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_whitespace_query_rejected(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/api/v1/rag/query",
            json={"query": "   \n\t  "},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_top_k_rejected(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/api/v1/rag/query",
            json={"query": "Valid query", "top_k": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422

        response = client.post(
            "/api/v1/rag/query",
            json={"query": "Valid query", "top_k": 100},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_similarity_threshold_rejected(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/api/v1/rag/query",
            json={"query": "Valid query", "similarity_threshold": 1.5},
            headers=auth_headers,
        )
        assert response.status_code == 422

        response = client.post(
            "/api/v1/rag/query",
            json={"query": "Valid query", "similarity_threshold": -0.1},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_authenticated_rag_query_success(self, client: TestClient, auth_headers: dict, test_user: User):
        mock_rag_service = MagicMock(spec=RAGService)
        mock_rag_service.generate_answer.return_value = RAGResponse(
            query="What is the refund deadline?",
            answer="According to the policy, all refunds must be requested within 30 days.",
            retrieved_context_count=2,
            model_name="gemini-2.5-flash",
        )

        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        try:
            response = client.post(
                "/api/v1/rag/query",
                json={
                    "query": "What is the refund deadline?",
                    "top_k": 3,
                    "similarity_threshold": 0.6,
                },
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "What is the refund deadline?"
            assert data["answer"] == "According to the policy, all refunds must be requested within 30 days."
            assert data["retrieved_context_count"] == 2
            assert data["model_name"] == "gemini-2.5-flash"

            mock_rag_service.generate_answer.assert_called_once_with(
                query="What is the refund deadline?",
                user_id=str(test_user.id),
                top_k=3,
                similarity_threshold=0.6,
                document_id=None,
                conversation_id=None,
            )
        finally:
            app.dependency_overrides.pop(get_rag_service, None)

    def test_rag_validation_error_mapped_to_422(self, client: TestClient, auth_headers: dict):
        mock_rag_service = MagicMock(spec=RAGService)
        mock_rag_service.generate_answer.side_effect = RAGValidationError("Invalid search query.")

        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        try:
            response = client.post(
                "/api/v1/rag/query",
                json={"query": "Valid length query"},
                headers=auth_headers,
            )
            assert response.status_code == 422
            assert "Invalid search query." in response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_rag_service, None)

    def test_rag_service_error_mapped_to_502(self, client: TestClient, auth_headers: dict):
        mock_rag_service = MagicMock(spec=RAGService)
        mock_rag_service.generate_answer.side_effect = RAGError("Answer generation failed: Gemini timeout")

        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        try:
            response = client.post(
                "/api/v1/rag/query",
                json={"query": "Valid query"},
                headers=auth_headers,
            )
            assert response.status_code == 502
            assert "Grounded answer generation failed" in response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_rag_service, None)

    def test_rag_query_returns_structured_citations(self, client: TestClient, auth_headers: dict, test_user: User):
        mock_rag_service = MagicMock(spec=RAGService)
        mock_rag_service.generate_answer.return_value = RAGResponse(
            query="What is the policy?",
            answer="According to the policy [Source 1], refunds take 30 days.",
            retrieved_context_count=1,
            citations=[
                {
                    "citation_id": "source_1",
                    "document_id": "doc-uuid-1",
                    "chunk_id": "chunk-uuid-1",
                    "filename": "policy.pdf",
                    "page_number": 2,
                    "chunk_index": 0,
                }
            ],
            conversation_id="conv-uuid-1",
            message_id="msg-uuid-1",
            model_name="gemini-2.5-flash",
        )

        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        try:
            response = client.post(
                "/api/v1/rag/query",
                json={
                    "query": "What is the policy?",
                    "conversation_id": "conv-uuid-1",
                },
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["citations"]) == 1
            assert data["citations"][0]["citation_id"] == "source_1"
            assert data["citations"][0]["filename"] == "policy.pdf"
            assert data["citations"][0]["page_number"] == 2
            assert data["conversation_id"] == "conv-uuid-1"
            assert data["message_id"] == "msg-uuid-1"
        finally:
            app.dependency_overrides.pop(get_rag_service, None)

    def test_rag_conversation_not_found_mapped_to_404(self, client: TestClient, auth_headers: dict):
        mock_rag_service = MagicMock(spec=RAGService)
        mock_rag_service.generate_answer.side_effect = RAGValidationError("Conversation 'non-existent' not found or unauthorized.")

        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        try:
            response = client.post(
                "/api/v1/rag/query",
                json={"query": "Valid query", "conversation_id": "non-existent"},
                headers=auth_headers,
            )
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_rag_service, None)
