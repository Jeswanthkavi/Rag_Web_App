"""Message ORM Model."""

from enum import Enum
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, ForeignKey, JSON, DateTime, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.models.conversation import Conversation


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MessageRole(str, Enum):
    """Role associated with a conversational turn."""
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class Message(Base):
    """An individual conversational turn containing message content and citation telemetry."""
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    citations: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON,
        nullable=True,
    )
    token_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("id", str(uuid.uuid4()))
        kwargs.setdefault("created_at", utc_now())
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<Message id={self.id} conv_id={self.conversation_id} role={self.role}>"
