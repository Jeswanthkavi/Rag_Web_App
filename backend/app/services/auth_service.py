"""Authentication Service containing business logic for registration, authentication, and token issuance."""

from typing import Optional
from fastapi import HTTPException, status
from backend.app.core.config import get_settings
from backend.app.core.security import hash_password, verify_password, create_access_token
from backend.app.models.user import User
from backend.app.repositories.user_repository import UserRepository
from backend.app.schemas.user import UserCreate
from backend.app.schemas.auth import Token


class AuthService:
    """Service encapsulating authentication workflows."""

    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def register_user(self, user_in: UserCreate) -> User:
        """Registers a new tenant user after validating email uniqueness."""
        existing = self.user_repo.get_by_email(user_in.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email address already exists.",
            )
            
        hashed_pw = hash_password(user_in.password)
        return self.user_repo.create(
            email=user_in.email,
            hashed_password=hashed_pw,
            full_name=user_in.full_name,
        )

    def authenticate_user(self, email: str, password: str) -> User:
        """Authenticates user credentials.
        
        Rejects invalid credentials generically without disclosing whether email or password was wrong.
        """
        user = self.user_repo.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive.",
            )
            
        return user

    def create_user_token(self, user: User) -> Token:
        """Issues a signed JWT access token for the authenticated user."""
        settings = get_settings()
        token_str = create_access_token(
            subject=user.id,
            email=user.email,
        )
        return Token(
            access_token=token_str,
            token_type="bearer",
            expires_in_seconds=settings.auth.access_token_expire_minutes * 60,
        )
