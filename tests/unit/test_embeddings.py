"""Unit tests for embedding providers, base validations, configuration, and EmbeddingService."""

import pytest
import math
from unittest.mock import MagicMock, patch
from typing import List

from backend.app.embeddings.base import BaseEmbeddingProvider, EmbeddingProviderError
from backend.app.embeddings.config import EmbeddingConfig
from backend.app.embeddings.local_provider import LocalEmbeddingProvider
from backend.app.embeddings.models import ChunkEmbedding, EmbeddingMetadata
from backend.app.embeddings.service import EmbeddingService
from backend.app.models.document_chunk import DocumentChunk


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic in-memory embedding provider for fast offline unit tests."""

    def __init__(self, dimension: int = 4, model_name: str = "fake-embed-v1"):
        self._dimension = dimension
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        return vectors[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for text in texts:
            # Deterministic pseudo-vector derived from string length and hash
            val = float(len(text) % 10)
            vec = [val + float(i) for i in range(self._dimension)]
            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        self.validate_vectors(vectors, len(texts))
        return vectors


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig validation and defaults."""

    def test_default_config_valid(self):
        cfg = EmbeddingConfig()
        assert cfg.provider == "local"
        assert cfg.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.dimension == 384
        assert cfg.batch_size == 32
        assert cfg.normalize_embeddings is True

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="provider cannot be empty"):
            EmbeddingConfig(provider="")

        with pytest.raises(ValueError, match="model_name cannot be empty"):
            EmbeddingConfig(model_name="")

        with pytest.raises(ValueError, match="dimension must be positive"):
            EmbeddingConfig(dimension=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingConfig(batch_size=-5)


class TestBaseEmbeddingProviderValidation:
    """Tests vector structure and dimension validation rules."""

    def test_validate_vector_success(self):
        provider = FakeEmbeddingProvider(dimension=3)
        assert provider.validate_vector([0.1, 0.2, 0.3]) is True

    def test_validate_vector_dimension_mismatch(self):
        provider = FakeEmbeddingProvider(dimension=3)
        with pytest.raises(EmbeddingProviderError, match="dimension mismatch"):
            provider.validate_vector([0.1, 0.2])

    def test_validate_vector_non_finite_values(self):
        provider = FakeEmbeddingProvider(dimension=3)
        with pytest.raises(EmbeddingProviderError, match="non-finite"):
            provider.validate_vector([0.1, float("nan"), 0.3])

        with pytest.raises(EmbeddingProviderError, match="non-finite"):
            provider.validate_vector([0.1, float("inf"), 0.3])

    def test_validate_vectors_batch_count_mismatch(self):
        provider = FakeEmbeddingProvider(dimension=2)
        with pytest.raises(EmbeddingProviderError, match="Batch vector count mismatch"):
            provider.validate_vectors([[0.1, 0.2]], expected_count=2)


class TestFakeProviderDeterministicBehavior:
    """Tests deterministic batch generation and ordering."""

    def test_empty_input_returns_empty(self):
        provider = FakeEmbeddingProvider()
        assert provider.embed_texts([]) == []

    def test_single_and_batch_consistency(self):
        provider = FakeEmbeddingProvider(dimension=4)
        v1 = provider.embed_text("Hello World")
        v2 = provider.embed_texts(["Hello World"])[0]
        assert v1 == v2
        assert len(v1) == 4

    def test_batch_preserves_positional_ordering(self):
        provider = FakeEmbeddingProvider(dimension=4)
        texts = ["Alpha", "Beta gamma", "Delta epsilon zeta"]
        vectors = provider.embed_texts(texts)
        assert len(vectors) == 3
        # Check that each vector equals single embedding
        for t, v in zip(texts, vectors):
            assert provider.embed_text(t) == v


class TestEmbeddingService:
    """Tests EmbeddingService orchestration, provider resolution, and chunk transformation."""

    def test_service_with_injected_provider(self):
        provider = FakeEmbeddingProvider(dimension=8, model_name="mock-model")
        service = EmbeddingService(provider=provider)

        assert service.provider_name == "fake"
        assert service.model_name == "mock-model"
        assert service.dimension == 8

        meta = service.get_metadata()
        assert meta.provider == "fake"
        assert meta.model_name == "mock-model"
        assert meta.dimension == 8

    def test_service_embed_chunks(self):
        provider = FakeEmbeddingProvider(dimension=4, model_name="fake-model")
        service = EmbeddingService(provider=provider)

        chunks = [
            DocumentChunk(
                id="c1-uuid",
                document_id="doc-123",
                user_id="user-456",
                chunk_index=0,
                chunk_text="Chunk zero text content.",
                token_count=4,
            ),
            DocumentChunk(
                id="c2-uuid",
                document_id="doc-123",
                user_id="user-456",
                chunk_index=1,
                chunk_text="Chunk one text content.",
                token_count=4,
            ),
        ]

        results: List[ChunkEmbedding] = service.embed_chunks(chunks)
        assert len(results) == 2

        # Check result 0
        assert results[0].chunk_id == "c1-uuid"
        assert results[0].chunk_index == 0
        assert results[0].document_id == "doc-123"
        assert len(results[0].vector) == 4
        assert results[0].dimension == 4
        assert results[0].model_name == "fake-model"

        # Check result 1
        assert results[1].chunk_id == "c2-uuid"
        assert results[1].chunk_index == 1
        assert results[1].document_id == "doc-123"
        assert len(results[1].vector) == 4

    def test_service_embed_empty_chunks(self):
        service = EmbeddingService(provider=FakeEmbeddingProvider())
        assert service.embed_chunks([]) == []

    def test_service_unknown_provider_raises(self):
        cfg = EmbeddingConfig(provider="unsupported-provider-xyz")
        with pytest.raises(EmbeddingProviderError, match="Unsupported embedding provider"):
            EmbeddingService(config=cfg)


class TestLocalEmbeddingProviderUnit:
    """Unit tests for LocalEmbeddingProvider with mocked SentenceTransformer."""

    def test_local_provider_lazy_loading(self):
        cfg = EmbeddingConfig(model_name="mock-transformer-model", dimension=4)
        provider = LocalEmbeddingProvider(config=cfg)
        assert provider._model is None

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            vec = provider.embed_text("Sample input text")
            assert len(vec) == 4
            assert provider._model is not None

    def test_local_provider_model_load_failure(self):
        cfg = EmbeddingConfig(model_name="nonexistent-model-12345")
        provider = LocalEmbeddingProvider(config=cfg)

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.side_effect = RuntimeError("Download Failed")

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            with pytest.raises(EmbeddingProviderError, match="Failed to load embedding model"):
                provider.embed_text("Test")
