"""API integration tests for /api/v1/retrieval/search endpoint."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.main import app
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.models.document import Document, DocumentStatus
from backend.app.retrieval.models import RetrievalResponse, RetrievalResult
from backend.app.retrieval.service import RetrievalService
from backend.app.api.dependencies import get_retrieval_service, get_current_active_user


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
        email="searcher@example.com",
        hashed_password="hashed_secure_password",
        full_name="Search Tester",
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


class TestRetrievalAPI:
    """Tests for POST /api/v1/retrieval/search."""

    def test_unauthenticated_search_rejected(self, client: TestClient):
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "Neural vector search", "top_k": 5},
        )
        assert response.status_code == 401

    def test_empty_query_rejected(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "", "top_k": 5},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_whitespace_query_rejected(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "   \n\t  ", "top_k": 5},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_top_k_rejected(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "Valid query", "top_k": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422

        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "Valid query", "top_k": 1000},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_similarity_threshold_rejected(self, client: TestClient, auth_headers: dict):
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "Valid query", "similarity_threshold": 1.5},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_authenticated_search_success(self, client: TestClient, auth_headers: dict, test_user: User):
        mock_retrieval_service = MagicMock(spec=RetrievalService)
        mock_retrieval_service.search.return_value = RetrievalResponse(
            query="neural search",
            results=[
                RetrievalResult(
                    chunk_id="chunk-1-uuid",
                    document_id="doc-1-uuid",
                    user_id=str(test_user.id),
                    chunk_index=0,
                    chunk_text="Neural search leverages dense vector representations.",
                    page_number=1,
                    filename="search.pdf",
                    file_type=".pdf",
                    token_count=12,
                    score=0.92,
                    rank=1,
                )
            ],
            total_retrieved=1,
            top_k=5,
            similarity_threshold=0.5,
        )

        app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_service
        try:
            response = client.post(
                "/api/v1/retrieval/search",
                json={"query": "neural search", "top_k": 5, "similarity_threshold": 0.5},
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "neural search"
            assert data["total_retrieved"] == 1
            assert len(data["results"]) == 1
            assert data["results"][0]["chunk_id"] == "chunk-1-uuid"
            assert data["results"][0]["score"] == 0.92
            assert data["results"][0]["rank"] == 1
            assert data["results"][0]["filename"] == "search.pdf"

            mock_retrieval_service.search.assert_called_once_with(
                query="neural search",
                user_id=str(test_user.id),
                top_k=5,
                similarity_threshold=0.5,
                document_id=None,
            )
        finally:
            app.dependency_overrides.pop(get_retrieval_service, None)
