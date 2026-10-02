"""Service orchestrating conversation lifecycle, tenant isolation, and bounded context assembly."""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from backend.app.models.conversation import Conversation
from backend.app.models.message import Message, MessageRole
from backend.app.repositories.conversation_repository import ConversationRepository
from backend.app.repositories.message_repository import MessageRepository

logger = logging.getLogger("app.services.conversation")


class ConversationServiceError(Exception):
    """Base exception for conversation service operations."""
    pass


class ConversationNotFoundError(ConversationServiceError):
    """Exception raised when a conversation is not found or not owned by the tenant."""
    pass


class ConversationService:
    """Business service managing conversations, messages, and prompt history formatting."""

    def __init__(
        self,
        db: Session,
        conv_repo: Optional[ConversationRepository] = None,
        msg_repo: Optional[MessageRepository] = None,
    ):
        self.db = db
        self.conv_repo = conv_repo or ConversationRepository(db)
        self.msg_repo = msg_repo or MessageRepository(db)

    def create_conversation(self, user_id: str, title: str = "New Conversation") -> Conversation:
        """Creates a new conversation session for an authenticated user."""
        return self.conv_repo.create(user_id=user_id, title=title)

    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        """Retrieves a conversation owned strictly by user_id."""
        return self.conv_repo.get_by_id(conversation_id=conversation_id, user_id=user_id)

    def list_conversations(self, user_id: str, skip: int = 0, limit: int = 50) -> List[Conversation]:
        """Lists user's conversations with pagination."""
        return self.conv_repo.list_by_user(user_id=user_id, skip=skip, limit=limit)

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Deletes a user-owned conversation."""
        return self.conv_repo.delete(conversation_id=conversation_id, user_id=user_id)

    def update_title(self, conversation_id: str, user_id: str, title: str) -> Optional[Conversation]:
        """Updates the conversation title if owned by the user."""
        return self.conv_repo.update_title(conversation_id=conversation_id, user_id=user_id, title=title)

    def get_conversation_with_messages(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        """Retrieves conversation and its message history, ensuring tenant isolation."""
        conversation = self.get_conversation(conversation_id=conversation_id, user_id=user_id)
        if not conversation:
            return None
        # Load messages explicitly
        conversation.messages = self.msg_repo.list_by_conversation(conversation_id=conversation_id)
        return conversation

    def get_bounded_history_for_prompt(
        self,
        conversation_id: str,
        user_id: str,
        max_messages: int = 10,
    ) -> str:
        """Retrieves and formats the latest `max_messages` turns chronologically for LLM context.
        
        Enforces tenant authorization before reading messages.
        """
        conversation = self.get_conversation(conversation_id=conversation_id, user_id=user_id)
        if not conversation:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found or unauthorized.")

        recent_messages = self.msg_repo.get_recent_messages(
            conversation_id=conversation_id,
            limit=max_messages,
        )

        if not recent_messages:
            return ""

        formatted_turns: List[str] = []
        for msg in recent_messages:
            role_prefix = "User" if msg.role == MessageRole.USER else "Assistant"
            formatted_turns.append(f"{role_prefix}: {msg.content.strip()}")

        return "\n\n".join(formatted_turns)

    def record_user_message(self, conversation_id: str, user_id: str, content: str) -> Message:
        """Records a user query turn within the conversation after verifying ownership."""
        conversation = self.get_conversation(conversation_id=conversation_id, user_id=user_id)
        if not conversation:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found or unauthorized.")

        return self.msg_repo.create(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
        )

    def record_assistant_message(
        self,
        conversation_id: str,
        user_id: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        token_metrics: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Records an assistant response turn within the conversation after verifying ownership."""
        conversation = self.get_conversation(conversation_id=conversation_id, user_id=user_id)
        if not conversation:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found or unauthorized.")

        return self.msg_repo.create(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=content,
            citations=citations,
            token_metrics=token_metrics,
        )
