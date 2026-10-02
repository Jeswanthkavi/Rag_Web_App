"""Repository for Message database operations."""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, desc, asc
from backend.app.models.message import Message, MessageRole

logger = logging.getLogger("app.repositories.message")


class MessageRepository:
    """Handles persistence and retrieval of conversational messages."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        token_metrics: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Persists a new message turn within a conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations,
            token_metrics=token_metrics,
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_recent_messages(self, conversation_id: str, limit: int = 10) -> List[Message]:
        """Fetches the most recent `limit` messages for a conversation, ordered chronologically (oldest to newest).
        
        Uses a subquery or reverse ordering to avoid loading entire conversation histories.
        """
        # Fetch newest `limit` messages descending by created_at
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(desc(Message.created_at), desc(Message.id))
            .limit(limit)
        )
        descending_messages = list(self.db.execute(stmt).scalars().all())
        # Reverse to chronological order (oldest to newest) for prompt context
        return list(reversed(descending_messages))

    def list_by_conversation(self, conversation_id: str) -> List[Message]:
        """Lists all messages for a conversation ordered chronologically."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(asc(Message.created_at), asc(Message.id))
        )
        return list(self.db.execute(stmt).scalars().all())
