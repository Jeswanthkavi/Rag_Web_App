"""Unit tests for semantic retrieval models, query validation, configuration, and RetrievalService."""

import pytest
from unittest.mock import MagicMock, patch
from typing import List

from backend.app.embeddings.base import EmbeddingProviderError
from backend.app.embeddings.service import EmbeddingService
from backend.app.retrieval.config import RetrievalConfig
from backend.app.retrieval.models import (
    RetrievalError,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
    RetrievalValidationError,
)
from backend.app.retrieval.service import RetrievalService
from backend.app.vectorstore.models import VectorStoreError
from tests.unit.test_embeddings import FakeEmbeddingProvider


class TestRetrievalConfig:
    """Tests for RetrievalConfig validation and defaults."""

    def test_default_config_valid(self):
        cfg = RetrievalConfig()
        assert cfg.top_k == 5
        assert cfg.similarity_threshold == 0.5
        assert cfg.max_query_length == 1000

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            RetrievalConfig(top_k=0)

        with pytest.raises(ValueError, match="top_k cannot exceed 100"):
            RetrievalConfig(top_k=101)

    def test_invalid_similarity_threshold_raises(self):
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            RetrievalConfig(similarity_threshold=-0.1)

        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            RetrievalConfig(similarity_threshold=1.5)

    def test_invalid_max_query_length_raises(self):
        with pytest.raises(ValueError, match="max_query_length must be between 10 and 10000"):
            RetrievalConfig(max_query_length=5)

        with pytest.raises(ValueError, match="max_query_length must be between 10 and 10000"):
            RetrievalConfig(max_query_length=20000)


class TestRetrievalQueryModel:
    """Tests for RetrievalQuery Pydantic request schema validation."""

    def test_valid_query_model(self):
        q = RetrievalQuery(query="What is neural search?", top_k=10, similarity_threshold=0.6)
        assert q.query == "What is neural search?"
        assert q.top_k == 10
        assert q.similarity_threshold == 0.6
        assert q.document_id is None

    def test_empty_query_raises_validation_error(self):
        with pytest.raises(ValueError):
            RetrievalQuery(query="")

    def test_invalid_top_k_bounds(self):
        with pytest.raises(ValueError):
            RetrievalQuery(query="Valid query", top_k=0)
        with pytest.raises(ValueError):
            RetrievalQuery(query="Valid query", top_k=51)

    def test_invalid_threshold_bounds(self):
        with pytest.raises(ValueError):
            RetrievalQuery(query="Valid query", similarity_threshold=-0.1)
        with pytest.raises(ValueError):
            RetrievalQuery(query="Valid query", similarity_threshold=1.1)


class TestRetrievalServiceUnit:
    """Unit tests for RetrievalService business logic, validation, error handling, and transformation."""

    def test_validate_query_trims_whitespace(self):
        service = RetrievalService(
            vector_repo=MagicMock(),
            embedding_service=EmbeddingService(provider=FakeEmbeddingProvider()),
            config=RetrievalConfig(),
        )
        res = service.validate_query("  What is semantic search?   \n")
        assert res == "What is semantic search?"

    def test_validate_query_empty_or_whitespace_raises(self):
        service = RetrievalService(
            vector_repo=MagicMock(),
            embedding_service=EmbeddingService(provider=FakeEmbeddingProvider()),
            config=RetrievalConfig(),
        )
        with pytest.raises(RetrievalValidationError, match="cannot be empty or whitespace-only"):
            service.validate_query("")

        with pytest.raises(RetrievalValidationError, match="cannot be empty or whitespace-only"):
            service.validate_query("   \t \n  ")

        with pytest.raises(RetrievalValidationError, match="cannot be None"):
            service.validate_query(None)

    def test_validate_query_exceeds_max_length_raises(self):
        service = RetrievalService(
            vector_repo=MagicMock(),
            embedding_service=EmbeddingService(provider=FakeEmbeddingProvider()),
            config=RetrievalConfig(max_query_length=50),
        )
        with pytest.raises(RetrievalValidationError, match="exceeds the maximum allowed length"):
            service.validate_query("A" * 51)

    def test_search_missing_user_id_raises(self):
        service = RetrievalService(
            vector_repo=MagicMock(),
            embedding_service=EmbeddingService(provider=FakeEmbeddingProvider()),
            config=RetrievalConfig(),
        )
        with pytest.raises(RetrievalValidationError, match="user_id is required"):
            service.search(query="Valid query", user_id="")

        with pytest.raises(RetrievalValidationError, match="user_id is required"):
            service.search(query="Valid query", user_id="   ")

    def test_search_query_embedding_service_called_with_sanitized_text(self):
        mock_repo = MagicMock()
        mock_repo.search.return_value = []
        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.1] * 384

        service = RetrievalService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
            config=RetrievalConfig(),
        )

        service.search(query="  Find documents about embeddings   ", user_id="user-123")
        mock_embedding.embed_text.assert_called_once_with("Find documents about embeddings")

    def test_search_tenant_and_document_filter_passed_to_repo(self):
        mock_repo = MagicMock()
        mock_repo.search.return_value = []
        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.1] * 384

        service = RetrievalService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
            config=RetrievalConfig(top_k=7, similarity_threshold=0.4),
        )

        service.search(
            query="Testing search",
            user_id="tenant-abc",
            document_id="doc-xyz",
            top_k=3,
            similarity_threshold=0.6,
        )

        mock_repo.search.assert_called_once_with(
            query_vector=[0.1] * 384,
            user_id="tenant-abc",
            document_id="doc-xyz",
            top_k=3,
            score_threshold=0.6,
            collection_name=service.config.collection_name,
        )

    def test_search_result_transformation_and_ranking(self):
        mock_repo = MagicMock()
        mock_repo.search.return_value = [
            {
                "id": "point-2",
                "score": 0.85,
                "payload": {
                    "chunk_id": "c2-uuid",
                    "document_id": "doc-1",
                    "user_id": "user-test",
                    "chunk_index": 1,
                    "chunk_text": "Second chunk content",
                    "page_number": 2,
                    "filename": "guide.pdf",
                    "file_type": ".pdf",
                    "token_count": 50,
                },
            },
            {
                "id": "point-1",
                "score": 0.95,
                "payload": {
                    "chunk_id": "c1-uuid",
                    "document_id": "doc-1",
                    "user_id": "user-test",
                    "chunk_index": 0,
                    "chunk_text": "First chunk content",
                    "page_number": 1,
                    "filename": "guide.pdf",
                    "file_type": ".pdf",
                    "token_count": 45,
                },
            },
        ]
        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.1] * 384

        service = RetrievalService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
            config=RetrievalConfig(top_k=5, similarity_threshold=0.5),
        )

        resp: RetrievalResponse = service.search(query="guide content", user_id="user-test")

        assert resp.query == "guide content"
        assert resp.total_retrieved == 2
        assert len(resp.results) == 2

        # Verify rank 1 is highest score (0.95)
        assert resp.results[0].rank == 1
        assert resp.results[0].chunk_id == "c1-uuid"
        assert resp.results[0].score == 0.95
        assert resp.results[0].chunk_text == "First chunk content"
        assert resp.results[0].filename == "guide.pdf"

        # Verify rank 2 is next highest score (0.85)
        assert resp.results[1].rank == 2
        assert resp.results[1].chunk_id == "c2-uuid"
        assert resp.results[1].score == 0.85

    def test_search_empty_results_handled_cleanly(self):
        mock_repo = MagicMock()
        mock_repo.search.return_value = []
        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.1] * 384

        service = RetrievalService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
            config=RetrievalConfig(),
        )

        resp = service.search(query="nonexistent term", user_id="user-empty")
        assert resp.total_retrieved == 0
        assert resp.results == []

    def test_search_score_threshold_filtering(self):
        mock_repo = MagicMock()
        # Mock repo returning items above and below threshold
        mock_repo.search.return_value = [
            {
                "id": "p1",
                "score": 0.75,
                "payload": {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "user_id": "u1",
                    "chunk_index": 0,
                    "chunk_text": "High score match",
                    "filename": "f.txt",
                    "file_type": ".txt",
                },
            },
            {
                "id": "p2",
                "score": 0.45,  # Below threshold 0.6
                "payload": {
                    "chunk_id": "c2",
                    "document_id": "d1",
                    "user_id": "u1",
                    "chunk_index": 1,
                    "chunk_text": "Low score match",
                    "filename": "f.txt",
                    "file_type": ".txt",
                },
            },
        ]
        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.1] * 384

        service = RetrievalService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
            config=RetrievalConfig(),
        )

        resp = service.search(query="test", user_id="u1", similarity_threshold=0.6)
        assert resp.total_retrieved == 1
        assert resp.results[0].chunk_id == "c1"
        assert resp.results[0].score == 0.75
        assert resp.results[0].rank == 1

    def test_search_malformed_payload_skipped_safely(self):
        mock_repo = MagicMock()
        mock_repo.search.return_value = [
            {
                "id": "p-malformed",
                "score": 0.9,
                "payload": {
                    # Missing chunk_id, document_id, chunk_text
                    "other_field": "data"
                },
            },
            {
                "id": "p-valid",
                "score": 0.8,
                "payload": {
                    "chunk_id": "c-valid",
                    "document_id": "doc-valid",
                    "user_id": "u-test",
                    "chunk_index": 0,
                    "chunk_text": "Valid text",
                    "filename": "valid.pdf",
                    "file_type": ".pdf",
                },
            },
        ]
        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.1] * 384

        service = RetrievalService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
            config=RetrievalConfig(similarity_threshold=0.1),
        )

        resp = service.search(query="test", user_id="u-test")
        assert resp.total_retrieved == 1
        assert resp.results[0].chunk_id == "c-valid"

    def test_search_tenant_anomaly_in_payload_skipped(self):
        mock_repo = MagicMock()
        mock_repo.search.return_value = [
            {
                "id": "p-wrong-user",
                "score": 0.99,
                "payload": {
                    "chunk_id": "c-wrong",
                    "document_id": "doc-other",
                    "user_id": "attacker-id",  # Mismatches querying user u-legit
                    "chunk_text": "Forbidden data",
                    "filename": "secret.pdf",
                    "file_type": ".pdf",
                },
            }
        ]
        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.1] * 384

        service = RetrievalService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
            config=RetrievalConfig(similarity_threshold=0.1),
        )

        resp = service.search(query="secret", user_id="u-legit")
        assert resp.total_retrieved == 0
        assert resp.results == []

    def test_search_embedding_failure_raises_retrieval_error(self):
        mock_repo = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embed_text.side_effect = EmbeddingProviderError("Model Timeout")

        service = RetrievalService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
            config=RetrievalConfig(),
        )

        with pytest.raises(RetrievalError, match="Query embedding failed"):
            service.search(query="query text", user_id="user-1")

    def test_search_qdrant_failure_raises_retrieval_error(self):
        mock_repo = MagicMock()
        mock_repo.search.side_effect = VectorStoreError("Connection Refused")
        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.1] * 384

        service = RetrievalService(
            vector_repo=mock_repo,
            embedding_service=mock_embedding,
            config=RetrievalConfig(),
        )

        with pytest.raises(RetrievalError, match="Semantic search failed"):
            service.search(query="query text", user_id="user-1")
