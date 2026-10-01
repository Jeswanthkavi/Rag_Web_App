"""User Pydantic schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserBase(BaseModel):
    """Base user properties."""
    email: EmailStr = Field(description="User unique email address")
    full_name: Optional[str] = Field(default=None, max_length=255, description="Full name of the user")


class UserCreate(UserBase):
    """User registration request schema."""
    password: str = Field(
        min_length=8,
        max_length=128,
        description="Plaintext password (minimum 8 characters)"
    )


class UserUpdate(BaseModel):
    """User profile update schema."""
    full_name: Optional[str] = Field(default=None, max_length=255)
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)


class UserResponse(UserBase):
    """Safe public user response schema.
    
    Excludes hashed_password and any internal security keys.
    """
    id: str = Field(description="Unique user ID (UUID)")
    is_active: bool = Field(description="User active account status")
    is_superuser: bool = Field(description="Superuser status flag")
    created_at: datetime = Field(description="Account creation timestamp (UTC)")
    updated_at: datetime = Field(description="Account last updated timestamp (UTC)")

    model_config = ConfigDict(from_attributes=True)
