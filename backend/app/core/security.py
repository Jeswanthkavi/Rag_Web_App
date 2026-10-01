"""Security utilities for password hashing and JWT token handling.

Uses Argon2id for password hashing and PyJWT for signed, expiring access tokens.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from backend.app.core.config import get_settings

# Argon2id password hasher instance
_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Hashes a plaintext password using Argon2id."""
    if not password:
        raise ValueError("Password cannot be empty")
    return _hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against an Argon2id hash.
    
    Fails safely without leaking information.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return _hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    except Exception:
        return False


def create_access_token(
    subject: str,
    email: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a signed JWT access token for the given subject (user ID)."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.auth.access_token_expire_minutes)
        
    payload: Dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    
    if email:
        payload["email"] = email
        
    if extra_claims:
        for k, v in extra_claims.items():
            if k not in payload:
                payload[k] = v
                
    secret = settings.auth.jwt_secret_key.get_secret_value()
    algorithm = settings.auth.jwt_algorithm
    
    encoded_jwt = jwt.encode(payload, secret, algorithm=algorithm)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and validates a signed JWT access token.
    
    Raises:
        jwt.ExpiredSignatureError: If the token has expired.
        jwt.InvalidTokenError: If the signature or claims are invalid.
    """
    settings = get_settings()
    secret = settings.auth.jwt_secret_key.get_secret_value()
    algorithm = settings.auth.jwt_algorithm
    
    decoded = jwt.decode(
        token,
        secret,
        algorithms=[algorithm],
        options={"require": ["exp", "sub"]},
    )
    return decoded
