"""Repository for Conversation database operations."""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from backend.app.models.conversation import Conversation

logger = logging.getLogger("app.repositories.conversation")


class ConversationRepository:
    """Handles tenant-isolated database access for Conversation models."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, user_id: str, title: str = "New Conversation") -> Conversation:
        """Creates and persists a new conversation session for a user."""
        conversation = Conversation(
            user_id=user_id,
            title=title.strip() if title else "New Conversation",
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get_by_id(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        """Retrieves a conversation strictly scoped to the authenticated user."""
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_by_user(self, user_id: str, skip: int = 0, limit: int = 50) -> List[Conversation]:
        """Lists conversations owned by the user, ordered by most recently updated."""
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def update_title(self, conversation_id: str, user_id: str, title: str) -> Optional[Conversation]:
        """Updates the conversation title if owned by the user."""
        conversation = self.get_by_id(conversation_id, user_id)
        if not conversation:
            return None
        conversation.title = title.strip()
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def delete(self, conversation_id: str, user_id: str) -> bool:
        """Deletes a conversation and cascade-deletes messages if owned by user."""
        conversation = self.get_by_id(conversation_id, user_id)
        if not conversation:
            return False
        self.db.delete(conversation)
        self.db.commit()
        return True
