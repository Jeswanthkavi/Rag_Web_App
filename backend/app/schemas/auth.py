"""Authentication Pydantic schemas."""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """User credentials for authentication."""
    email: EmailStr = Field(description="User email address")
    password: str = Field(min_length=1, description="User password")


class Token(BaseModel):
    """JWT Bearer token response."""
    access_token: str = Field(description="Signed JWT access token")
    token_type: str = Field(default="bearer", description="Token type, standard bearer")
    expires_in_seconds: int = Field(description="Token validity in seconds")


class TokenPayload(BaseModel):
    """Decoded JWT payload claims."""
    sub: str = Field(description="Subject identifier (user_id)")
    email: Optional[str] = Field(default=None, description="User email claim")
    exp: int = Field(description="Token expiration timestamp in UTC seconds")
