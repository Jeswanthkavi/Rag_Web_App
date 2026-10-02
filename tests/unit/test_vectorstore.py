"""Unit tests for VectorStore configuration, repository, and service with mocked Qdrant client."""

import pytest
from unittest.mock import MagicMock, patch
from typing import List

from qdrant_client.http import models as rest_models
from backend.app.embeddings.base import BaseEmbeddingProvider, EmbeddingProviderError
from backend.app.embeddings.service import EmbeddingService
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_chunk import DocumentChunk
from backend.app.models.user import User
from backend.app.vectorstore.config import VectorStoreConfig
from backend.app.vectorstore.models import (
    IndexingResult,
    VectorPoint,
    VectorStoreError,
    VectorStoreHealth,
)
from backend.app.vectorstore.repository import QdrantVectorRepository
from backend.app.vectorstore.service import VectorStoreService


class MockEmbeddingProvider(BaseEmbeddingProvider):
    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-embedding-model"

    @property
    def dimension(self) -> int:
        return 4

    def embed_text(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class TestVectorStoreConfig:
    """Tests configuration loading and constraints."""

    def test_default_config_valid(self):
        cfg = VectorStoreConfig()
        assert cfg.url == "http://localhost:6333"
        assert cfg.collection_name == "rag_document_chunks"
        assert cfg.dimension == 384
        assert cfg.batch_size == 64

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Qdrant URL cannot be empty"):
            VectorStoreConfig(url="")

        with pytest.raises(ValueError, match="collection_name cannot be empty"):
            VectorStoreConfig(collection_name="")

        with pytest.raises(ValueError, match="timeout must be positive"):
            VectorStoreConfig(timeout=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            VectorStoreConfig(batch_size=-10)

        with pytest.raises(ValueError, match="dimension must be positive"):
            VectorStoreConfig(dimension=0)


class TestQdrantVectorRepositoryUnit:
    """Unit tests for QdrantVectorRepository with mock client."""

    def test_initialize_collection_creates_when_missing(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        repo = QdrantVectorRepository(client=mock_client, config=VectorStoreConfig(dimension=384))
        result = repo.initialize_collection(collection_name="test_col")

        assert result is True
        mock_client.collection_exists.assert_called_once_with(collection_name="test_col")
        mock_client.create_collection.assert_called_once()

    def test_initialize_collection_verifies_existing_compatible(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True

        mock_info = MagicMock()
        mock_info.config.params.vectors = rest_models.VectorParams(
            size=384,
            distance=rest_models.Distance.COSINE,
        )
        mock_client.get_collection.return_value = mock_info

        repo = QdrantVectorRepository(client=mock_client, config=VectorStoreConfig(dimension=384))
        result = repo.initialize_collection(collection_name="test_col")

        assert result is True
        mock_client.create_collection.assert_not_called()

    def test_initialize_collection_incompatible_dimension_raises(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True

        mock_info = MagicMock()
        mock_info.config.params.vectors = rest_models.VectorParams(
            size=768,  # Mismatch
            distance=rest_models.Distance.COSINE,
        )
        mock_client.get_collection.return_value = mock_info

        repo = QdrantVectorRepository(client=mock_client, config=VectorStoreConfig(dimension=384))
        with pytest.raises(VectorStoreError, match="incompatible dimension"):
            repo.initialize_collection(collection_name="test_col")

    def test_upsert_points_batching(self):
        mock_client = MagicMock()
        repo = QdrantVectorRepository(client=mock_client, config=VectorStoreConfig(batch_size=2))

        points = [
            VectorPoint(id="p1", vector=[0.1, 0.2], payload={"text": "t1"}),
            VectorPoint(id="p2", vector=[0.3, 0.4], payload={"text": "t2"}),
            VectorPoint(id="p3", vector=[0.5, 0.6], payload={"text": "t3"}),
        ]

        upserted = repo.upsert_points(points, collection_name="test_col")
        assert upserted == 3
        assert mock_client.upsert.call_count == 2

    def test_get_document_point_ids_scrolling(self):
        mock_client = MagicMock()
        rec1 = MagicMock(id="p-1")
        rec2 = MagicMock(id="p-2")
        mock_client.scroll.return_value = ([rec1, rec2], None)

        repo = QdrantVectorRepository(client=mock_client)
        point_ids = repo.get_document_point_ids("doc-1", "user-1", "test_col")

        assert point_ids == ["p-1", "p-2"]
        mock_client.scroll.assert_called_once()

    def test_delete_points_by_ids(self):
        mock_client = MagicMock()
        repo = QdrantVectorRepository(client=mock_client)

        res = repo.delete_points(["p-1", "p-2"], "test_col")
        assert res == 2
        mock_client.delete.assert_called_once()

    def test_delete_by_document_tenant_filter(self):
        mock_client = MagicMock()
        repo = QdrantVectorRepository(client=mock_client)

        repo.delete_by_document(document_id="doc-123", user_id="user-456", collection_name="test_col")
        mock_client.delete.assert_called_once()
        call_kwargs = mock_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "test_col"
        filter_obj: rest_models.Filter = call_kwargs["points_selector"]
        assert len(filter_obj.must) == 2


class TestVectorStoreServiceUnit:
    """Unit tests for VectorStoreService orchestration and consistency."""

    def test_index_document_empty_chunks(self):
        mock_repo = MagicMock()
        mock_db = MagicMock()
        service = VectorStoreService(
            vector_repo=mock_repo,
            embedding_service=EmbeddingService(provider=MockEmbeddingProvider()),
        )

        doc = Document(id="doc-1", user_id="user-1", filename="empty.txt", file_type=".txt", file_size_bytes=0, storage_path="")
        
        with patch("backend.app.vectorstore.service.DocumentChunkRepository") as mock_chunk_repo_cls:
            mock_chunk_repo = MagicMock()
            mock_chunk_repo.get_by_document.return_value = []
            mock_chunk_repo_cls.return_value = mock_chunk_repo

            result = service.index_document(document=doc, db=mock_db)

            assert result.indexed_chunks_count == 0
            assert result.point_ids == []
            mock_repo.delete_by_document.assert_called_once_with(
                document_id="doc-1", user_id="user-1", collection_name=service.config.collection_name
            )

    def test_index_document_success_syncs_mysql(self):
        mock_repo = MagicMock()
        mock_repo.get_document_point_ids.return_value = []
        mock_repo.upsert_points.return_value = 2
        mock_db = MagicMock()

        service = VectorStoreService(
            vector_repo=mock_repo,
            embedding_service=EmbeddingService(provider=MockEmbeddingProvider()),
        )

        doc = Document(id="doc-1", user_id="user-1", filename="sample.pdf", file_type=".pdf", file_size_bytes=100, storage_path="")
        chunks = [
            DocumentChunk(id="c1-uuid", document_id="doc-1", user_id="user-1", chunk_index=0, chunk_text="Chunk 0", token_count=2),
            DocumentChunk(id="c2-uuid", document_id="doc-1", user_id="user-1", chunk_index=1, chunk_text="Chunk 1", token_count=2),
        ]

        with patch("backend.app.vectorstore.service.DocumentChunkRepository") as mock_chunk_repo_cls:
            mock_chunk_repo = MagicMock()
            mock_chunk_repo.get_by_document.return_value = chunks
            mock_chunk_repo_cls.return_value = mock_chunk_repo

            result = service.index_document(document=doc, db=mock_db)

            assert result.indexed_chunks_count == 2
            assert result.point_ids == ["c1-uuid", "c2-uuid"]
            assert chunks[0].qdrant_point_id == "c1-uuid"
            assert chunks[1].qdrant_point_id == "c2-uuid"
            mock_db.commit.assert_called_once()
            # Since no existing points, delete_points should not be called
            mock_repo.delete_points.assert_not_called()

    def test_index_document_stale_point_cleanup(self):
        """Verifies that only stale old point IDs are deleted after new points are upserted."""
        mock_repo = MagicMock()
        # Suppose Qdrant previously had points ['stale-1', 'stale-2', 'c1-uuid']
        mock_repo.get_document_point_ids.return_value = ["stale-1", "stale-2", "c1-uuid"]
        mock_repo.upsert_points.return_value = 2
        mock_db = MagicMock()

        service = VectorStoreService(
            vector_repo=mock_repo,
            embedding_service=EmbeddingService(provider=MockEmbeddingProvider()),
        )

        doc = Document(id="doc-1", user_id="user-1", filename="sample.pdf", file_type=".pdf", file_size_bytes=100, storage_path="")
        # Current chunks: 'c1-uuid' and 'c2-uuid'
        chunks = [
            DocumentChunk(id="c1-uuid", document_id="doc-1", user_id="user-1", chunk_index=0, chunk_text="Chunk 0", token_count=2),
            DocumentChunk(id="c2-uuid", document_id="doc-1", user_id="user-1", chunk_index=1, chunk_text="Chunk 1", token_count=2),
        ]

        with patch("backend.app.vectorstore.service.DocumentChunkRepository") as mock_chunk_repo_cls:
            mock_chunk_repo = MagicMock()
            mock_chunk_repo.get_by_document.return_value = chunks
            mock_chunk_repo_cls.return_value = mock_chunk_repo

            result = service.index_document(document=doc, db=mock_db)

            # Check that current points were upserted
            assert mock_repo.upsert_points.called
            # Check that ONLY stale-1 and stale-2 were deleted (not c1-uuid or c2-uuid)
            mock_repo.delete_points.assert_called_once()
            deleted_ids = mock_repo.delete_points.call_args[1]["point_ids"]
            assert set(deleted_ids) == {"stale-1", "stale-2"}
            assert "c1-uuid" not in deleted_ids
            assert "c2-uuid" not in deleted_ids

    def test_index_document_upsert_failure_preserves_existing_vectors(self):
        """Verifies that an upsert failure rolls back DB and does NOT delete any existing vectors."""
        mock_repo = MagicMock()
        mock_repo.get_document_point_ids.return_value = ["existing-1", "existing-2"]
        mock_repo.upsert_points.side_effect = RuntimeError("Qdrant Disk Full")
        mock_db = MagicMock()

        service = VectorStoreService(
            vector_repo=mock_repo,
            embedding_service=EmbeddingService(provider=MockEmbeddingProvider()),
        )

        doc = Document(id="doc-1", user_id="user-1", filename="sample.pdf", file_type=".pdf", file_size_bytes=100, storage_path="")
        chunks = [
            DocumentChunk(id="c1-uuid", document_id="doc-1", user_id="user-1", chunk_index=0, chunk_text="Chunk 0", token_count=2),
        ]

        with patch("backend.app.vectorstore.service.DocumentChunkRepository") as mock_chunk_repo_cls:
            mock_chunk_repo = MagicMock()
            mock_chunk_repo.get_by_document.return_value = chunks
            mock_chunk_repo_cls.return_value = mock_chunk_repo

            with pytest.raises(VectorStoreError, match="Document indexing failed"):
                service.index_document(document=doc, db=mock_db)

            # Invariant: delete_points and delete_by_document must NOT have been called!
            mock_repo.delete_points.assert_not_called()
            mock_repo.delete_by_document.assert_not_called()
            mock_db.rollback.assert_called_once()

    def test_index_document_embedding_failure(self):
        """Verifies that an embedding generation failure aborts before any Qdrant write."""
        mock_repo = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.dimension = 4
        mock_embedding.embed_chunks.side_effect = EmbeddingProviderError("Model Timeout")
        mock_db = MagicMock()

        service = VectorStoreService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
        )

        doc = Document(id="doc-1", user_id="user-1", filename="sample.pdf", file_type=".pdf", file_size_bytes=100, storage_path="")
        chunks = [
            DocumentChunk(id="c1-uuid", document_id="doc-1", user_id="user-1", chunk_index=0, chunk_text="Chunk 0", token_count=2),
        ]

        with patch("backend.app.vectorstore.service.DocumentChunkRepository") as mock_chunk_repo_cls:
            mock_chunk_repo = MagicMock()
            mock_chunk_repo.get_by_document.return_value = chunks
            mock_chunk_repo_cls.return_value = mock_chunk_repo

            with pytest.raises(VectorStoreError, match="Document indexing failed"):
                service.index_document(document=doc, db=mock_db)

            mock_repo.upsert_points.assert_not_called()
            mock_repo.delete_points.assert_not_called()
            mock_db.rollback.assert_called_once()
