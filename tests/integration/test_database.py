"""Integration tests for MySQL database schema and SQLAlchemy ORM models (Phase 3)."""

from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from backend.app.db.session import check_db_health
from backend.app.models import (
    User,
    Document,
    DocumentStatus,
    DocumentChunk,
    Conversation,
    Message,
    MessageRole,
)


class TestDatabaseConnectivity:
    """Test connection pooling and health checks on the test database."""

    def test_database_health_check(self, db_session: Session):
        health = check_db_health(db_session)
        assert health["status"] == "healthy"
        assert health["is_alive"] is True
        assert health["database_name"] == "advanced_rag_test"
        assert health["version"] is not None


class TestUserEntityAndConstraints:
    """Test User entity persistence and constraints."""

    def test_create_user_success(self, db_session: Session):
        user = User(
            email="researcher@example.com",
            hashed_password="$2b$12$securehashvalue1234567890",
            full_name="Lead Researcher",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        assert user.id is not None
        assert user.email == "researcher@example.com"
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_duplicate_email_raises_integrity_error(self, db_session: Session):
        user1 = User(
            email="unique@example.com",
            hashed_password="hash_value_1",
            full_name="User One",
        )
        db_session.add(user1)
        db_session.commit()

        user2 = User(
            email="unique@example.com",
            hashed_password="hash_value_2",
            full_name="User Two",
        )
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


class TestDocumentAndChunkRelationships:
    """Test Document and DocumentChunk entity persistence and cascade relationships."""

    def test_create_document_and_chunks(self, db_session: Session):
        # Create user
        user = User(
            email="docowner@example.com",
            hashed_password="some_password_hash",
            full_name="Document Owner",
        )
        db_session.add(user)
        db_session.commit()

        # Create document
        doc = Document(
            user_id=user.id,
            filename="quarterly_report.pdf",
            file_type=".pdf",
            file_size_bytes=1048576,
            storage_path="./storage/uploads/user1/quarterly_report.pdf",
            status=DocumentStatus.PROCESSING,
            page_count=12,
        )
        db_session.add(doc)
        db_session.commit()

        # Add chunks
        chunk1 = DocumentChunk(
            document_id=doc.id,
            user_id=user.id,
            chunk_index=0,
            chunk_text="Executive Summary of Q3 Financial Performance.",
            page_number=1,
            token_count=15,
            qdrant_point_id="point-1-uuid",
        )
        chunk2 = DocumentChunk(
            document_id=doc.id,
            user_id=user.id,
            chunk_index=1,
            chunk_text="Revenue increased by 22% year over year.",
            page_number=1,
            token_count=18,
            qdrant_point_id="point-2-uuid",
        )
        db_session.add_all([chunk1, chunk2])
        doc.total_chunks = 2
        doc.status = DocumentStatus.COMPLETED
        db_session.commit()

        # Query back and verify relationships
        saved_doc = db_session.execute(select(Document).where(Document.id == doc.id)).scalar_one()
        assert saved_doc.owner.email == "docowner@example.com"
        assert len(saved_doc.chunks) == 2
        assert saved_doc.chunks[0].chunk_text == "Executive Summary of Q3 Financial Performance."
        assert saved_doc.chunks[1].qdrant_point_id == "point-2-uuid"

    def test_delete_document_cascades_to_chunks(self, db_session: Session):
        user = User(
            email="cascade_doc@example.com",
            hashed_password="hash",
            full_name="Cascade User",
        )
        db_session.add(user)
        db_session.commit()

        doc = Document(
            user_id=user.id,
            filename="to_delete.pdf",
            file_type=".pdf",
            file_size_bytes=5000,
            storage_path="./storage/to_delete.pdf",
            status=DocumentStatus.COMPLETED,
        )
        db_session.add(doc)
        db_session.commit()

        chunk = DocumentChunk(
            document_id=doc.id,
            user_id=user.id,
            chunk_index=0,
            chunk_text="Chunk to be cascade deleted.",
            page_number=1,
        )
        db_session.add(chunk)
        db_session.commit()

        chunk_id = chunk.id
        # Delete document
        db_session.delete(doc)
        db_session.commit()

        # Verify chunk was cascade deleted
        retrieved_chunk = db_session.execute(select(DocumentChunk).where(DocumentChunk.id == chunk_id)).scalar_one_or_none()
        assert retrieved_chunk is None


class TestConversationAndMessageRelationships:
    """Test Conversation and Message persistence with JSON metadata and cascades."""

    def test_create_conversation_with_messages(self, db_session: Session):
        user = User(
            email="chat_user@example.com",
            hashed_password="hash",
            full_name="Chat User",
        )
        db_session.add(user)
        db_session.commit()

        conv = Conversation(
            user_id=user.id,
            title="Q3 Strategy Q&A",
        )
        db_session.add(conv)
        db_session.commit()

        now = datetime.now(timezone.utc)
        msg_user = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="What was the net revenue growth?",
            created_at=now,
        )
        msg_assistant = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content="Revenue grew by 22% in Q3 [1].",
            citations=[{
                "document_id": "doc-uuid-1",
                "filename": "quarterly_report.pdf",
                "page": 1,
                "chunk_index": 1,
                "score": 0.92,
            }],
            token_metrics={
                "prompt_tokens": 120,
                "completion_tokens": 35,
                "latency_ms": 340,
            },
            created_at=now + timedelta(seconds=1),
        )
        db_session.add_all([msg_user, msg_assistant])
        db_session.commit()

        saved_conv = db_session.execute(select(Conversation).where(Conversation.id == conv.id)).scalar_one()
        assert len(saved_conv.messages) == 2
        
        user_messages = [m for m in saved_conv.messages if m.role == MessageRole.USER]
        assistant_messages = [m for m in saved_conv.messages if m.role == MessageRole.ASSISTANT]
        
        assert len(user_messages) == 1
        assert len(assistant_messages) == 1
        assert user_messages[0].content == "What was the net revenue growth?"
        assert assistant_messages[0].citations[0]["filename"] == "quarterly_report.pdf"
        assert assistant_messages[0].token_metrics["latency_ms"] == 340

    def test_delete_conversation_cascades_to_messages(self, db_session: Session):
        user = User(
            email="conv_del@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        db_session.commit()

        conv = Conversation(user_id=user.id, title="Temporary Conversation")
        db_session.add(conv)
        db_session.commit()

        msg = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="Hello world",
        )
        db_session.add(msg)
        db_session.commit()

        msg_id = msg.id
        # Delete conversation
        db_session.delete(conv)
        db_session.commit()

        # Verify message was cascade deleted
        retrieved_msg = db_session.execute(select(Message).where(Message.id == msg_id)).scalar_one_or_none()
        assert retrieved_msg is None


class TestUserCascadeTenancyDeletion:
    """Verify deleting a User cascades and purges all documents, chunks, and conversations."""

    def test_delete_user_purges_all_tenant_entities(self, db_session: Session):
        user = User(
            email="tenant_purge@example.com",
            hashed_password="hash",
            full_name="Tenant to Delete",
        )
        db_session.add(user)
        db_session.commit()

        doc = Document(
            user_id=user.id,
            filename="tenant_file.pdf",
            file_type=".pdf",
            file_size_bytes=100,
            storage_path="./path.pdf",
        )
        db_session.add(doc)
        db_session.commit()

        chunk = DocumentChunk(
            document_id=doc.id,
            user_id=user.id,
            chunk_index=0,
            chunk_text="Tenant text chunk",
        )
        conv = Conversation(user_id=user.id, title="Tenant Conversation")
        db_session.add_all([chunk, conv])
        db_session.commit()

        msg = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="Tenant message",
        )
        db_session.add(msg)
        db_session.commit()

        doc_id = doc.id
        chunk_id = chunk.id
        conv_id = conv.id
        msg_id = msg.id

        # Delete user
        db_session.delete(user)
        db_session.commit()

        # Verify all child entities are removed
        assert db_session.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none() is None
        assert db_session.execute(select(DocumentChunk).where(DocumentChunk.id == chunk_id)).scalar_one_or_none() is None
        assert db_session.execute(select(Conversation).where(Conversation.id == conv_id)).scalar_one_or_none() is None
        assert db_session.execute(select(Message).where(Message.id == msg_id)).scalar_one_or_none() is None
