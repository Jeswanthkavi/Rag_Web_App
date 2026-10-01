"""User Repository for abstracted database operations on User entities."""

from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.models.user import User


class UserRepository:
    """Repository handling persistence operations for User models."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: str) -> Optional[User]:
        """Finds a user by primary key ID."""
        statement = select(User).where(User.id == user_id)
        return self.db.execute(statement).scalar_one_or_none()

    def get_by_email(self, email: str) -> Optional[User]:
        """Finds a user by normalized email address."""
        normalized_email = email.strip().lower()
        statement = select(User).where(User.email == normalized_email)
        return self.db.execute(statement).scalar_one_or_none()

    def create(
        self,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
        is_active: bool = True,
        is_superuser: bool = False,
    ) -> User:
        """Persists a new user record."""
        user = User(
            email=email.strip().lower(),
            hashed_password=hashed_password,
            full_name=full_name.strip() if full_name else None,
            is_active=is_active,
            is_superuser=is_superuser,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update(
        self,
        user: User,
        full_name: Optional[str] = None,
        hashed_password: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> User:
        """Updates user profile attributes."""
        if full_name is not None:
            user.full_name = full_name.strip()
        if hashed_password is not None:
            user.hashed_password = hashed_password
        if is_active is not None:
            user.is_active = is_active
            
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user: User) -> None:
        """Deletes a user record and cascades to tenant entities."""
        self.db.delete(user)
        self.db.commit()
