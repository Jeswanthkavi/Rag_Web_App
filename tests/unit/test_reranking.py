"""Unit tests for Phase 16 Cross-Encoder Reranking.

Tests verify:
1. Reranker receives dense candidates
2. Reranker changes ordering correctly based on relevance score
3. Final top-K is strictly respected (slicing)
4. Candidate pool: retrieval_top_k > final_top_k
5. Dense score preserved alongside rerank score
6. Rerank score preserved
7. Reranker failure falls back to dense order
8. Empty candidate set handling
9. Single candidate handling
10. Tenant isolation preserved
11. Deterministic FakeReranker behavior with custom scores / custom functions
12. LocalReranker lazy loading, batching, and error handling
13. RerankingConfig validation
14. RerankingService disabled mode
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.app.reranking.base import (
    BaseReranker,
    RerankerError,
)
from backend.app.reranking.config import RerankingConfig
from backend.app.reranking.fake_provider import FakeReranker
from backend.app.reranking.local_provider import LocalReranker
from backend.app.reranking.models import RerankResponse, RerankResult
from backend.app.reranking.service import RerankingService
from backend.app.retrieval.models import RetrievalResult


def _make_candidate(
    chunk_id: str,
    chunk_text: str,
    score: float = 0.8,
    rank: int = 1,
    user_id: str = "user-1",
    filename: str = "doc.txt",
    page_number: int = 1,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        user_id=user_id,
        chunk_index=0,
        chunk_text=chunk_text,
        page_number=page_number,
        filename=filename,
        file_type=".txt",
        token_count=len(chunk_text.split()),
        score=score,
        rank=rank,
    )


class TestRerankingConfig:
    """Test RerankingConfig validation rules."""

    def test_default_config_valid(self):
        cfg = RerankingConfig()
        assert cfg.enabled is True
        assert cfg.retrieval_top_k == 20
        assert cfg.final_top_k == 5
        assert cfg.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.batch_size == 32

    def test_final_top_k_exceeding_retrieval_top_k_rejected(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            RerankingConfig(retrieval_top_k=5, final_top_k=10)

    def test_invalid_top_k_rejected(self):
        with pytest.raises(ValueError, match="retrieval_top_k"):
            RerankingConfig(retrieval_top_k=0)
        with pytest.raises(ValueError, match="final_top_k"):
            RerankingConfig(final_top_k=0)


class TestFakeReranker:
    """Test deterministic mock reranker behavior."""

    def test_empty_candidates_returns_empty_list(self):
        reranker = FakeReranker()
        results = reranker.rerank(query="test", candidates=[])
        assert results == []

    def test_reordering_candidates_by_custom_scores(self):
        candidates = [
            _make_candidate("c1", "Low relevance chunk", score=0.9, rank=1),
            _make_candidate("c2", "Highly relevant battery warranty chunk", score=0.7, rank=2),
            _make_candidate("c3", "Medium relevance chunk", score=0.8, rank=3),
        ]

        reranker = FakeReranker(
            custom_scores={"c1": 1.0, "c2": 9.5, "c3": 5.0}
        )
        results = reranker.rerank(query="battery warranty", candidates=candidates)

        assert len(results) == 3
        # c2 should now be ranked #1
        assert results[0].chunk_id == "c2"
        assert results[0].rerank_score == 9.5
        assert results[0].dense_score == 0.7
        assert results[0].rank == 1

        # c3 should be #2
        assert results[1].chunk_id == "c3"
        assert results[1].rerank_score == 5.0
        assert results[1].dense_score == 0.8
        assert results[1].rank == 2

        # c1 should be #3
        assert results[2].chunk_id == "c1"
        assert results[2].rerank_score == 1.0
        assert results[2].dense_score == 0.9
        assert results[2].rank == 3

    def test_reverse_order_reranking(self):
        candidates = [
            _make_candidate("c1", "First chunk", score=0.9, rank=1),
            _make_candidate("c2", "Second chunk", score=0.8, rank=2),
            _make_candidate("c3", "Third chunk", score=0.7, rank=3),
        ]

        reranker = FakeReranker(reverse_order=True)
        results = reranker.rerank(query="any", candidates=candidates)

        assert [r.chunk_id for r in results] == ["c3", "c2", "c1"]
        assert [r.rank for r in results] == [1, 2, 3]

    def test_simulated_failure_raises_error(self):
        candidates = [_make_candidate("c1", "Text", score=0.8)]
        reranker = FakeReranker(should_fail=True)
        with pytest.raises(RerankerError, match="Simulated"):
            reranker.rerank(query="query", candidates=candidates)


class TestRerankingService:
    """Test service orchestration, slicing, fallbacks, and tenant protection."""

    def test_reranking_service_reorders_and_slices_top_k(self):
        # Pool of 5 candidates -> slice to top 2
        candidates = [
            _make_candidate("c1", "General overview", score=0.95, rank=1),
            _make_candidate("c2", "The battery is covered under 2-year warranty", score=0.85, rank=2),
            _make_candidate("c3", "Shipping policy", score=0.80, rank=3),
            _make_candidate("c4", "Battery replacement cost is $50", score=0.75, rank=4),
            _make_candidate("c5", "Terms of service", score=0.70, rank=5),
        ]

        fake_reranker = FakeReranker(
            custom_scores={"c1": 1.0, "c2": 10.0, "c3": 0.5, "c4": 8.0, "c5": 0.1}
        )
        config = RerankingConfig(retrieval_top_k=5, final_top_k=2)
        service = RerankingService(reranker=fake_reranker, config=config)

        results = service.rerank(query="battery warranty", candidates=candidates, top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "c2"
        assert results[0].rank == 1
        assert results[0].dense_score == 0.85
        assert results[0].rerank_score == 10.0

        assert results[1].chunk_id == "c4"
        assert results[1].rank == 2
        assert results[1].dense_score == 0.75
        assert results[1].rerank_score == 8.0

    def test_reranker_failure_falls_back_to_dense_order(self):
        candidates = [
            _make_candidate("c1", "First dense candidate", score=0.95, rank=1),
            _make_candidate("c2", "Second dense candidate", score=0.85, rank=2),
            _make_candidate("c3", "Third dense candidate", score=0.75, rank=3),
        ]

        fake_reranker = FakeReranker(should_fail=True)
        config = RerankingConfig(final_top_k=2)
        service = RerankingService(reranker=fake_reranker, config=config)

        # Must not raise exception; falls back to top 2 dense candidates in dense order
        results = service.rerank(query="query", candidates=candidates, top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[0].dense_score == 0.95
        assert results[0].rank == 1

        assert results[1].chunk_id == "c2"
        assert results[1].dense_score == 0.85
        assert results[1].rank == 2

    def test_disabled_reranking_preserves_dense_order(self):
        candidates = [
            _make_candidate("c1", "Candidate 1", score=0.9, rank=1),
            _make_candidate("c2", "Candidate 2", score=0.8, rank=2),
        ]
        fake_reranker = FakeReranker()
        config = RerankingConfig(enabled=False, final_top_k=2)
        service = RerankingService(reranker=fake_reranker, config=config)

        results = service.rerank(query="query", candidates=candidates)

        assert len(results) == 2
        assert fake_reranker.call_count == 0  # Bypassed
        assert results[0].chunk_id == "c1"
        assert results[0].dense_score == 0.9
        assert results[1].chunk_id == "c2"
        assert results[1].dense_score == 0.8

    def test_empty_candidate_list_handled_safely(self):
        service = RerankingService(reranker=FakeReranker())
        results = service.rerank(query="query", candidates=[])
        assert results == []

    def test_single_candidate_handled(self):
        candidates = [_make_candidate("c1", "Only candidate", score=0.88)]
        service = RerankingService(reranker=FakeReranker(custom_scores={"c1": 5.0}))
        results = service.rerank(query="query", candidates=candidates)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].rank == 1
        assert results[0].rerank_score == 5.0
        assert results[0].dense_score == 0.88

    def test_tenant_isolation_preserved(self):
        """Verify reranker only evaluates the candidates passed to it from tenant-filtered retrieval."""
        user1_candidates = [
            _make_candidate("u1_c1", "User 1 secret", user_id="user-1"),
        ]
        reranker = FakeReranker()
        service = RerankingService(reranker=reranker)

        results = service.rerank(query="secret", candidates=user1_candidates)
        assert len(results) == 1
        assert results[0].user_id == "user-1"


class TestLocalReranker:
    """Test LocalReranker provider mechanics with mocked CrossEncoder."""

    def test_local_reranker_batches_and_predicts(self):
        candidates = [
            _make_candidate("c1", "Text chunk 1", score=0.9),
            _make_candidate("c2", "Text chunk 2", score=0.8),
        ]

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.2, 0.9]  # c2 scored higher

        reranker = LocalReranker()
        reranker._model = mock_cross_encoder

        results = reranker.rerank(query="test query", candidates=candidates, top_k=2)

        # Verify prediction called with pair list
        mock_cross_encoder.predict.assert_called_once_with(
            [["test query", "Text chunk 1"], ["test query", "Text chunk 2"]],
            batch_size=32,
            show_progress_bar=False,
        )

        assert len(results) == 2
        # c2 has score 0.9 -> rank 1
        assert results[0].chunk_id == "c2"
        assert results[0].rerank_score == 0.9
        assert results[0].dense_score == 0.8
        assert results[0].rank == 1

        # c1 has score 0.2 -> rank 2
        assert results[1].chunk_id == "c1"
        assert results[1].rerank_score == 0.2
        assert results[1].dense_score == 0.9
        assert results[1].rank == 2

    def test_local_reranker_exception_mapped_to_reranker_error(self):
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.side_effect = RuntimeError("GPU out of memory")

        reranker = LocalReranker()
        reranker._model = mock_cross_encoder

        candidates = [_make_candidate("c1", "Text", score=0.9)]
        with pytest.raises(RerankerError, match="Reranking execution failed"):
            reranker.rerank(query="query", candidates=candidates)
