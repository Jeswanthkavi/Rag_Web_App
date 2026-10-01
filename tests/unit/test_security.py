"""Unit tests for Argon2id password hashing and JWT security functions (Phase 5)."""

from datetime import timedelta
import pytest
import jwt
from backend.app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)


class TestPasswordHashing:
    """Test Argon2id password hashing and verification."""

    def test_hash_password_returns_argon2_hash(self):
        raw_pw = "superSecretPassword123!"
        hashed = hash_password(raw_pw)
        
        assert hashed != raw_pw
        assert hashed.startswith("$argon2id$")
        assert verify_password(raw_pw, hashed) is True

    def test_verify_password_mismatch(self):
        hashed = hash_password("correctPassword")
        assert verify_password("wrongPassword", hashed) is False
        assert verify_password("", hashed) is False
        assert verify_password("correctPassword", "") is False
        assert verify_password("correctPassword", "not_a_valid_argon2_hash") is False

    def test_hash_empty_password_raises_error(self):
        with pytest.raises(ValueError):
            hash_password("")


class TestJWTTokens:
    """Test JWT creation, decoding, expiration, and tampering detection."""

    def test_create_and_decode_valid_token(self):
        user_id = "user-uuid-12345"
        email = "test@example.com"
        token = create_access_token(subject=user_id, email=email)
        
        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # Header, Payload, Signature
        
        payload = decode_access_token(token)
        assert payload["sub"] == user_id
        assert payload["email"] == email
        assert "exp" in payload
        assert "iat" in payload

    def test_decode_expired_token_raises_error(self):
        user_id = "user-uuid-12345"
        # Create token that expired 10 minutes ago
        token = create_access_token(
            subject=user_id,
            expires_delta=timedelta(minutes=-10),
        )
        
        with pytest.raises(jwt.ExpiredSignatureError):
            decode_access_token(token)

    def test_decode_tampered_token_raises_error(self):
        token = create_access_token(subject="user-123")
        tampered_token = token[:-5] + "XXXXX"
        
        with pytest.raises(jwt.InvalidTokenError):
            decode_access_token(tampered_token)

    def test_decode_garbage_string_raises_error(self):
        with pytest.raises(jwt.InvalidTokenError):
            decode_access_token("this-is-not-a-valid-jwt")
