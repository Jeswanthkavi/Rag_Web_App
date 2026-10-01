"""Integration tests for Authentication API endpoints (Phase 5)."""

from datetime import timedelta
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select
from backend.app.main import app
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.core.security import create_access_token, verify_password


@pytest.fixture
def client(db_session: Session) -> TestClient:
    """Provides a TestClient with database session overridden to advanced_rag_test."""
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestUserRegistrationAPI:
    """Test POST /api/v1/auth/register endpoint."""

    def test_register_user_success(self, client: TestClient, db_session: Session):
        payload = {
            "email": "engineer@example.com",
            "password": "SecurePassword123!",
            "full_name": "Senior Engineer",
        }
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 201
        data = response.json()
        
        # Verify returned user fields
        assert data["email"] == "engineer@example.com"
        assert data["full_name"] == "Senior Engineer"
        assert data["is_active"] is True
        assert data["is_superuser"] is False
        assert "id" in data
        assert "created_at" in data
        
        # Security assertion: Never expose password or hash in response
        assert "password" not in data
        assert "hashed_password" not in data
        
        # Verify DB storage: Password is an Argon2 hash, not plaintext
        db_user = db_session.execute(select(User).where(User.email == "engineer@example.com")).scalar_one()
        assert db_user.hashed_password != "SecurePassword123!"
        assert db_user.hashed_password.startswith("$argon2id$")
        assert verify_password("SecurePassword123!", db_user.hashed_password) is True

    def test_register_duplicate_email_fails(self, client: TestClient):
        payload = {
            "email": "duplicate@example.com",
            "password": "SecurePassword123!",
            "full_name": "First User",
        }
        # First registration
        res1 = client.post("/api/v1/auth/register", json=payload)
        assert res1.status_code == 201

        # Second registration with duplicate email
        res2 = client.post("/api/v1/auth/register", json=payload)
        assert res2.status_code == 409
        assert "already exists" in res2.json()["detail"].lower()

    def test_register_invalid_email_format(self, client: TestClient):
        payload = {
            "email": "invalid-email-string",
            "password": "SecurePassword123!",
        }
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 422

    def test_register_short_password_fails_validation(self, client: TestClient):
        payload = {
            "email": "shortpw@example.com",
            "password": "short",  # < 8 chars
        }
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 422


class TestUserLoginAPI:
    """Test POST /api/v1/auth/login endpoint."""

    def test_login_success(self, client: TestClient):
        # Register user first
        reg_payload = {
            "email": "login_user@example.com",
            "password": "ValidPassword999!",
            "full_name": "Login User",
        }
        reg_res = client.post("/api/v1/auth/register", json=reg_payload)
        assert reg_res.status_code == 201

        # Login with correct credentials
        login_payload = {
            "email": "login_user@example.com",
            "password": "ValidPassword999!",
        }
        response = client.post("/api/v1/auth/login", json=login_payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in_seconds"] > 0
        assert len(data["access_token"].split(".")) == 3

    def test_login_wrong_password_returns_generic_401(self, client: TestClient):
        # Register user
        reg_payload = {
            "email": "user_wrong_pw@example.com",
            "password": "CorrectPassword123!",
        }
        client.post("/api/v1/auth/register", json=reg_payload)

        # Login with wrong password
        login_payload = {
            "email": "user_wrong_pw@example.com",
            "password": "WrongPassword456!",
        }
        response = client.post("/api/v1/auth/login", json=login_payload)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid email or password."

    def test_login_nonexistent_email_returns_generic_401(self, client: TestClient):
        login_payload = {
            "email": "nonexistent_email@example.com",
            "password": "AnyPassword123!",
        }
        response = client.post("/api/v1/auth/login", json=login_payload)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid email or password."


class TestProtectedUserProfileAPI:
    """Test GET /api/v1/users/me endpoint."""

    def test_get_current_user_profile_success(self, client: TestClient):
        # Register and login
        reg_payload = {
            "email": "profile_user@example.com",
            "password": "SecurePassword123!",
            "full_name": "Profile Owner",
        }
        client.post("/api/v1/auth/register", json=reg_payload)
        
        login_res = client.post("/api/v1/auth/login", json={
            "email": "profile_user@example.com",
            "password": "SecurePassword123!",
        })
        token = login_res.json()["access_token"]

        # Request /users/me with Bearer token
        response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "profile_user@example.com"
        assert data["full_name"] == "Profile Owner"
        assert data["is_active"] is True
        
        # Security assertion: Never expose password or hash
        assert "password" not in data
        assert "hashed_password" not in data

    def test_get_current_user_missing_token_returns_401(self, client: TestClient):
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401
        assert "token required" in response.json()["detail"].lower()

    def test_get_current_user_invalid_token_returns_401(self, client: TestClient):
        response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": "Bearer invalid.fake.token"},
        )
        assert response.status_code == 401

    def test_get_current_user_expired_token_returns_401(self, client: TestClient, db_session: Session):
        # Create user in DB
        reg_payload = {
            "email": "expired_test@example.com",
            "password": "SecurePassword123!",
        }
        res = client.post("/api/v1/auth/register", json=reg_payload)
        user_id = res.json()["id"]

        # Create expired token
        expired_token = create_access_token(
            subject=user_id,
            expires_delta=timedelta(minutes=-5),
        )

        response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()

    def test_inactive_user_access_rejected(self, client: TestClient, db_session: Session):
        # Register user
        reg_payload = {
            "email": "inactive_user@example.com",
            "password": "SecurePassword123!",
        }
        res = client.post("/api/v1/auth/register", json=reg_payload)
        user_id = res.json()["id"]

        # Deactivate user in database
        user = db_session.execute(select(User).where(User.id == user_id)).scalar_one()
        user.is_active = False
        db_session.commit()

        # Generate valid token for this user
        token = create_access_token(subject=user_id)

        # Attempt to access /users/me
        response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "inactive" in response.json()["detail"].lower()
