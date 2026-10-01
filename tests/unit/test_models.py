"""Unit tests for SQLAlchemy models and mixins (Phase 3)."""

from backend.app.models.user import User
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_chunk import DocumentChunk
from backend.app.models.conversation import Conversation
from backend.app.models.message import Message, MessageRole


class TestModelInstantiations:
    """Test standard model instantiations, reprs, and default values."""

    def test_user_model_defaults(self):
        user = User(
            email="testuser@example.com",
            hashed_password="hashed_pw_test",
            full_name="Test User",
        )
        assert user.email == "testuser@example.com"
        assert user.is_active is True
        assert user.is_superuser is False
        assert "testuser@example.com" in repr(user)

    def test_document_model_defaults(self):
        doc = Document(
            user_id="user-123",
            filename="document.pdf",
            file_type=".pdf",
            file_size_bytes=1024,
            storage_path="./storage/uploads/user-123/doc.pdf",
        )
        assert doc.filename == "document.pdf"
        assert doc.status == DocumentStatus.PENDING
        assert doc.total_chunks == 0
        assert "document.pdf" in repr(doc)

    def test_document_chunk_model(self):
        chunk = DocumentChunk(
            document_id="doc-123",
            user_id="user-123",
            chunk_index=0,
            chunk_text="This is a test chunk text.",
            page_number=1,
            token_count=10,
            qdrant_point_id="point-uuid-123",
        )
        assert chunk.chunk_index == 0
        assert chunk.page_number == 1
        assert chunk.qdrant_point_id == "point-uuid-123"
        assert "doc-123" in repr(chunk)

    def test_conversation_model(self):
        conv = Conversation(
            user_id="user-123",
            title="Analysis Thread",
        )
        assert conv.title == "Analysis Thread"
        assert "Analysis Thread" in repr(conv)

    def test_message_model(self):
        msg = Message(
            conversation_id="conv-123",
            role=MessageRole.ASSISTANT,
            content="This is an assistant response.",
            citations=[{"document_id": "doc-1", "chunk_id": "chunk-1", "page": 2}],
            token_metrics={"prompt_tokens": 50, "completion_tokens": 20},
        )
        assert msg.role == MessageRole.ASSISTANT
        assert len(msg.citations) == 1
        assert msg.token_metrics["prompt_tokens"] == 50
        assert "conv-123" in repr(msg)
