"""API tests for /api/v1/conversations endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.main import app
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.models.conversation import Conversation
from backend.app.models.message import Message, MessageRole


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
def user_1(db_session: Session) -> User:
    user = User(
        email="conv_user_1@example.com",
        hashed_password="hashed_secure_password_1",
        full_name="Conversation User One",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def user_2(db_session: Session) -> User:
    user = User(
        email="conv_user_2@example.com",
        hashed_password="hashed_secure_password_2",
        full_name="Conversation User Two",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers_1(user_1: User) -> dict:
    from backend.app.core.security import create_access_token
    token = create_access_token(subject=str(user_1.id), email=user_1.email)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_2(user_2: User) -> dict:
    from backend.app.core.security import create_access_token
    token = create_access_token(subject=str(user_2.id), email=user_2.email)
    return {"Authorization": f"Bearer {token}"}


class TestConversationsAPI:
    """Tests for conversation management endpoints."""

    def test_unauthenticated_access_rejected(self, client: TestClient):
        response = client.get("/api/v1/conversations")
        assert response.status_code == 401

        response = client.post("/api/v1/conversations", json={"title": "Test"})
        assert response.status_code == 401

    def test_create_conversation_success(self, client: TestClient, auth_headers_1: dict, user_1: User):
        response = client.post(
            "/api/v1/conversations",
            json={"title": "Solar Research"},
            headers=auth_headers_1,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Solar Research"
        assert data["user_id"] == str(user_1.id)
        assert "id" in data

    def test_list_conversations_tenant_isolation(
        self,
        client: TestClient,
        auth_headers_1: dict,
        auth_headers_2: dict,
        db_session: Session,
        user_1: User,
        user_2: User,
    ):
        # Create convs for user 1
        conv1 = Conversation(user_id=str(user_1.id), title="User 1 Conv A")
        conv2 = Conversation(user_id=str(user_1.id), title="User 1 Conv B")
        # Create conv for user 2
        conv3 = Conversation(user_id=str(user_2.id), title="User 2 Secret")
        db_session.add_all([conv1, conv2, conv3])
        db_session.commit()

        # User 1 lists conversations
        res1 = client.get("/api/v1/conversations", headers=auth_headers_1)
        assert res1.status_code == 200
        items1 = res1.json()
        assert len(items1) == 2
        titles1 = [c["title"] for c in items1]
        assert "User 1 Conv A" in titles1
        assert "User 1 Conv B" in titles1
        assert "User 2 Secret" not in titles1

        # User 2 lists conversations
        res2 = client.get("/api/v1/conversations", headers=auth_headers_2)
        assert res2.status_code == 200
        items2 = res2.json()
        assert len(items2) == 1
        assert items2[0]["title"] == "User 2 Secret"

    def test_get_conversation_detail_with_messages(
        self,
        client: TestClient,
        auth_headers_1: dict,
        db_session: Session,
        user_1: User,
    ):
        conv = Conversation(user_id=str(user_1.id), title="Detail Test")
        db_session.add(conv)
        db_session.commit()
        db_session.refresh(conv)

        msg1 = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="Question turn",
        )
        msg2 = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content="Answer turn [Source 1]",
            citations=[{"citation_id": "source_1", "filename": "doc.pdf"}],
        )
        db_session.add_all([msg1, msg2])
        db_session.commit()

        response = client.get(f"/api/v1/conversations/{conv.id}", headers=auth_headers_1)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == conv.id
        assert data["title"] == "Detail Test"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "Question turn"
        assert data["messages"][0]["role"] == "USER"
        assert data["messages"][1]["content"] == "Answer turn [Source 1]"
        assert data["messages"][1]["role"] == "ASSISTANT"
        assert data["messages"][1]["citations"] is not None

    def test_cross_user_get_conversation_returns_404(
        self,
        client: TestClient,
        auth_headers_2: dict,
        db_session: Session,
        user_1: User,
    ):
        conv = Conversation(user_id=str(user_1.id), title="User 1 Private")
        db_session.add(conv)
        db_session.commit()
        db_session.refresh(conv)

        # User 2 attempts to get User 1's conversation
        response = client.get(f"/api/v1/conversations/{conv.id}", headers=auth_headers_2)
        assert response.status_code == 404

    def test_update_conversation_title(
        self,
        client: TestClient,
        auth_headers_1: dict,
        db_session: Session,
        user_1: User,
    ):
        conv = Conversation(user_id=str(user_1.id), title="Old Title")
        db_session.add(conv)
        db_session.commit()
        db_session.refresh(conv)

        response = client.patch(
            f"/api/v1/conversations/{conv.id}",
            json={"title": "New Title"},
            headers=auth_headers_1,
        )
        assert response.status_code == 200
        assert response.json()["title"] == "New Title"

    def test_delete_conversation_and_cascade(
        self,
        client: TestClient,
        auth_headers_1: dict,
        auth_headers_2: dict,
        db_session: Session,
        user_1: User,
    ):
        conv = Conversation(user_id=str(user_1.id), title="To Delete")
        db_session.add(conv)
        db_session.commit()
        db_session.refresh(conv)

        # User 2 cannot delete User 1's conversation
        res_cross = client.delete(f"/api/v1/conversations/{conv.id}", headers=auth_headers_2)
        assert res_cross.status_code == 404

        # User 1 deletes
        res = client.delete(f"/api/v1/conversations/{conv.id}", headers=auth_headers_1)
        assert res.status_code == 204

        # Subsequent get returns 404
        res_get = client.get(f"/api/v1/conversations/{conv.id}", headers=auth_headers_1)
        assert res_get.status_code == 404
