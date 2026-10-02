"""Unit tests for Conversation models, repositories, and ConversationService."""

import pytest
from sqlalchemy.orm import Session

from backend.app.models.user import User
from backend.app.models.conversation import Conversation
from backend.app.models.message import Message, MessageRole
from backend.app.repositories.conversation_repository import ConversationRepository
from backend.app.repositories.message_repository import MessageRepository
from backend.app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)


@pytest.fixture
def user_a(db_session: Session) -> User:
    user = User(
        email="user_a@example.com",
        hashed_password="hashed_pwd_a",
        full_name="User A",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def user_b(db_session: Session) -> User:
    user = User(
        email="user_b@example.com",
        hashed_password="hashed_pwd_b",
        full_name="User B",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


class TestConversationRepository:
    """Unit tests for ConversationRepository."""

    def test_create_and_get_conversation(self, db_session: Session, user_a: User):
        repo = ConversationRepository(db_session)
        conv = repo.create(user_id=str(user_a.id), title="Research Thread")

        assert conv.id is not None
        assert conv.user_id == str(user_a.id)
        assert conv.title == "Research Thread"

        fetched = repo.get_by_id(conv.id, str(user_a.id))
        assert fetched is not None
        assert fetched.id == conv.id

    def test_tenant_isolation_on_get(self, db_session: Session, user_a: User, user_b: User):
        repo = ConversationRepository(db_session)
        conv_a = repo.create(user_id=str(user_a.id), title="Private A")

        # User B should not be able to retrieve User A's conversation
        assert repo.get_by_id(conv_a.id, str(user_b.id)) is None

    def test_list_by_user(self, db_session: Session, user_a: User, user_b: User):
        repo = ConversationRepository(db_session)
        conv1 = repo.create(user_id=str(user_a.id), title="Thread 1")
        conv2 = repo.create(user_id=str(user_a.id), title="Thread 2")
        conv_b = repo.create(user_id=str(user_b.id), title="Thread B")

        list_a = repo.list_by_user(str(user_a.id))
        assert len(list_a) == 2
        titles_a = [c.title for c in list_a]
        assert "Thread 1" in titles_a
        assert "Thread 2" in titles_a
        assert "Thread B" not in titles_a

    def test_update_title(self, db_session: Session, user_a: User, user_b: User):
        repo = ConversationRepository(db_session)
        conv = repo.create(user_id=str(user_a.id), title="Original Title")

        # User B cannot update User A's conversation
        assert repo.update_title(conv.id, str(user_b.id), "Hacked Title") is None

        # User A updates successfully
        updated = repo.update_title(conv.id, str(user_a.id), "Updated Title")
        assert updated is not None
        assert updated.title == "Updated Title"

    def test_delete_conversation(self, db_session: Session, user_a: User, user_b: User):
        repo = ConversationRepository(db_session)
        conv = repo.create(user_id=str(user_a.id), title="To Delete")

        # User B cannot delete User A's conversation
        assert repo.delete(conv.id, str(user_b.id)) is False

        # User A deletes successfully
        assert repo.delete(conv.id, str(user_a.id)) is True
        assert repo.get_by_id(conv.id, str(user_a.id)) is None


class TestMessageRepository:
    """Unit tests for MessageRepository."""

    def test_create_and_fetch_messages(self, db_session: Session, user_a: User):
        conv_repo = ConversationRepository(db_session)
        msg_repo = MessageRepository(db_session)

        conv = conv_repo.create(user_id=str(user_a.id), title="Message Thread")

        msg1 = msg_repo.create(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="Hello assistant",
        )
        msg2 = msg_repo.create(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content="Hello user, how can I help?",
            citations=[{"citation_id": "source_1", "filename": "doc.pdf"}],
        )

        assert msg1.id is not None
        assert msg2.id is not None
        assert msg2.citations is not None
        assert len(msg2.citations) == 1

        all_msgs = msg_repo.list_by_conversation(conv.id)
        assert len(all_msgs) == 2
        assert all_msgs[0].content == "Hello assistant"
        assert all_msgs[1].content == "Hello user, how can I help?"

    def test_get_recent_messages_bounded_and_chronological(self, db_session: Session, user_a: User):
        conv_repo = ConversationRepository(db_session)
        msg_repo = MessageRepository(db_session)

        conv = conv_repo.create(user_id=str(user_a.id), title="Long Thread")

        for i in range(10):
            msg_repo.create(
                conversation_id=conv.id,
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=f"Message turn {i}",
            )

        # Request latest 4 messages
        recent = msg_repo.get_recent_messages(conv.id, limit=4)
        assert len(recent) == 4
        # Chronological order check: 6, 7, 8, 9
        assert recent[0].content == "Message turn 6"
        assert recent[1].content == "Message turn 7"
        assert recent[2].content == "Message turn 8"
        assert recent[3].content == "Message turn 9"


class TestConversationService:
    """Unit tests for ConversationService."""

    def test_get_bounded_history_for_prompt(self, db_session: Session, user_a: User):
        service = ConversationService(db_session)
        conv = service.create_conversation(user_id=str(user_a.id), title="Chat Thread")

        service.record_user_message(conv.id, str(user_a.id), "What is solar efficiency?")
        service.record_assistant_message(conv.id, str(user_a.id), "Solar efficiency is 22% [Source 1].")
        service.record_user_message(conv.id, str(user_a.id), "What about temperature limits?")

        history = service.get_bounded_history_for_prompt(conv.id, str(user_a.id), max_messages=2)
        assert "User: What is solar efficiency?" not in history  # truncated by limit=2
        assert "Assistant: Solar efficiency is 22% [Source 1]." in history
        assert "User: What about temperature limits?" in history

    def test_cross_tenant_access_raises_not_found(self, db_session: Session, user_a: User, user_b: User):
        service = ConversationService(db_session)
        conv_a = service.create_conversation(user_id=str(user_a.id), title="Private Thread")

        with pytest.raises(ConversationNotFoundError):
            service.get_bounded_history_for_prompt(conv_a.id, str(user_b.id))

        with pytest.raises(ConversationNotFoundError):
            service.record_user_message(conv_a.id, str(user_b.id), "Unauthorized message")

        with pytest.raises(ConversationNotFoundError):
            service.record_assistant_message(conv_a.id, str(user_b.id), "Unauthorized response")
