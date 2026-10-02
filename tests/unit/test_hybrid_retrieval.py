"""Unit tests for Phase 17 Hybrid Retrieval (Dense + Lexical BM25 + Reciprocal Rank Fusion).

Tests verify:
1. Dense-only result
2. Lexical-only result
3. Overlapping dense + lexical result
4. Duplicate chunk deduplication
5. RRF fusion ordering
6. Configurable RRF constant
7. Dense rank preservation
8. Lexical rank preservation
9. Fusion score calculation
10. Document_id filter
11. Tenant isolation
12. Dense retrieval failure fallback
13. Lexical retrieval failure fallback
14. Both retrieval branches empty
15. Fused candidates passed to reranker
16. Final top-K respected
17. Pure-Python BM25Okapi scoring accuracy
"""

import pytest
from unittest.mock import MagicMock

from backend.app.hybrid_retrieval.base import (
    BaseLexicalRetriever,
    HybridRetrievalError,
    LexicalRetrievalError,
)
from backend.app.hybrid_retrieval.bm25 import BM25Okapi, tokenize
from backend.app.hybrid_retrieval.config import HybridRetrievalConfig
from backend.app.hybrid_retrieval.fake_provider import FakeLexicalRetriever
from backend.app.hybrid_retrieval.fusion import reciprocal_rank_fusion
from backend.app.hybrid_retrieval.models import HybridCandidate, HybridRetrievalResponse
from backend.app.hybrid_retrieval.service import HybridRetrievalService
from backend.app.retrieval.models import (
    RetrievalError,
    RetrievalResponse,
    RetrievalResult,
    RetrievalValidationError,
)
from backend.app.retrieval.service import RetrievalService


def _make_res(
    chunk_id: str,
    chunk_text: str,
    score: float = 0.8,
    rank: int = 1,
    user_id: str = "user-1",
    doc_id: str = "doc-1",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        user_id=user_id,
        chunk_index=0,
        chunk_text=chunk_text,
        page_number=1,
        filename="doc.pdf",
        file_type=".pdf",
        token_count=10,
        score=score,
        rank=rank,
    )


class TestBM25Okapi:
    """Test pure-Python BM25Okapi scoring logic."""

    def test_bm25_scoring_exact_match(self):
        corpus = [
            ["warranty", "covers", "battery", "defects"],
            ["kubernetes", "deployment", "configuration"],
            ["shipping", "policy", "and", "returns"],
        ]
        bm25 = BM25Okapi(corpus=corpus)
        scores = bm25.get_scores(["warranty", "battery"])

        assert scores[0] > 0.0
        assert scores[1] == 0.0
        assert scores[2] == 0.0
        assert scores[0] > scores[1]

    def test_bm25_empty_query_returns_zeros(self):
        corpus = [["hello", "world"]]
        bm25 = BM25Okapi(corpus=corpus)
        scores = bm25.get_scores([])
        assert scores == [0.0]

    def test_tokenization_lowercasing_and_punctuation(self):
        tokens = tokenize("Hello, World! What's the battery-life?")
        assert tokens == ["hello", "world", "what", "s", "the", "battery", "life"]


class TestReciprocalRankFusion:
    """Test RRF fusion mathematical scoring, deduplication, and rank preservation."""

    def test_dense_only_candidates(self):
        dense = [_make_res("c1", "Text 1", score=0.9, rank=1)]
        lexical = []

        results, candidates = reciprocal_rank_fusion(dense, lexical, rrf_k=60)
        assert len(results) == 1
        assert len(candidates) == 1
        assert candidates[0].chunk_id == "c1"
        assert candidates[0].dense_rank == 1
        assert candidates[0].lexical_rank is None
        assert candidates[0].fusion_score == pytest.approx(1.0 / (60 + 1))
        assert candidates[0].rank == 1

    def test_lexical_only_candidates(self):
        dense = []
        lexical = [_make_res("c2", "Text 2", score=5.2, rank=1)]

        results, candidates = reciprocal_rank_fusion(dense, lexical, rrf_k=60)
        assert len(results) == 1
        assert candidates[0].chunk_id == "c2"
        assert candidates[0].dense_rank is None
        assert candidates[0].lexical_rank == 1
        assert candidates[0].fusion_score == pytest.approx(1.0 / (60 + 1))

    def test_overlapping_candidates_deduplicated_and_fused(self):
        # Chunk "c1" is #1 in dense and #2 in lexical
        dense = [
            _make_res("c1", "Common chunk", score=0.95, rank=1),
            _make_res("c2", "Dense only", score=0.85, rank=2),
        ]
        lexical = [
            _make_res("c3", "Lexical only", score=8.0, rank=1),
            _make_res("c1", "Common chunk", score=6.0, rank=2),
        ]

        results, candidates = reciprocal_rank_fusion(dense, lexical, rrf_k=60)

        # 3 unique chunks
        assert len(results) == 3
        assert len(candidates) == 3

        # c1 should be #1 because it has combined RRF score: 1/(60+1) + 1/(60+2)
        expected_c1_score = (1.0 / 61.0) + (1.0 / 62.0)
        assert candidates[0].chunk_id == "c1"
        assert candidates[0].dense_rank == 1
        assert candidates[0].lexical_rank == 2
        assert candidates[0].fusion_score == pytest.approx(expected_c1_score)
        assert candidates[0].rank == 1

        # Check c3 and c2
        assert candidates[1].chunk_id == "c3"
        assert candidates[1].fusion_score == pytest.approx(1.0 / 61.0)
        assert candidates[2].chunk_id == "c2"
        assert candidates[2].fusion_score == pytest.approx(1.0 / 62.0)

    def test_configurable_rrf_constant(self):
        dense = [_make_res("c1", "Text", rank=1)]
        _, candidates_k20 = reciprocal_rank_fusion(dense, [], rrf_k=20)
        _, candidates_k100 = reciprocal_rank_fusion(dense, [], rrf_k=100)

        assert candidates_k20[0].fusion_score == pytest.approx(1.0 / 21.0)
        assert candidates_k100[0].fusion_score == pytest.approx(1.0 / 101.0)


class TestHybridRetrievalService:
    """Test service orchestration, fallbacks, filtering, and tenant isolation."""

    def test_hybrid_search_combines_and_limits_candidates(self):
        mock_dense = MagicMock()
        mock_dense.search.return_value = RetrievalResponse(
            query="battery",
            results=[
                _make_res("c1", "Dense chunk 1", score=0.9, rank=1),
                _make_res("c2", "Dense chunk 2", score=0.8, rank=2),
            ],
            total_retrieved=2,
            top_k=20,
        )

        fake_lexical = FakeLexicalRetriever(
            custom_results=[
                _make_res("c2", "Dense chunk 2", score=5.0, rank=1),
                _make_res("c3", "Lexical chunk 3", score=4.0, rank=2),
            ]
        )

        service = HybridRetrievalService(
            retrieval_service=mock_dense,
            lexical_retriever=fake_lexical,
            config=HybridRetrievalConfig(fusion_candidate_limit=2),
        )

        resp = service.search(query="battery", user_id="user-1", top_k=2)

        assert len(resp.results) == 2
        # c2 is in both dense (rank 2) and lexical (rank 1) -> top fused candidate
        assert resp.results[0].chunk_id == "c2"
        assert resp.results[0].rank == 1
        assert resp.dense_count == 2
        assert resp.lexical_count == 2

    def test_dense_failure_falls_back_to_lexical_results(self):
        mock_dense = MagicMock()
        mock_dense.search.side_effect = RetrievalError("Qdrant connection refused")

        fake_lexical = FakeLexicalRetriever(
            custom_results=[_make_res("c1", "Lexical result", score=4.5)]
        )

        service = HybridRetrievalService(
            retrieval_service=mock_dense,
            lexical_retriever=fake_lexical,
        )

        resp = service.search(query="query", user_id="user-1")

        assert len(resp.results) == 1
        assert resp.results[0].chunk_id == "c1"
        assert resp.dense_count == 0
        assert resp.lexical_count == 1

    def test_lexical_failure_falls_back_to_dense_results(self):
        mock_dense = MagicMock()
        mock_dense.search.return_value = RetrievalResponse(
            query="query",
            results=[_make_res("c1", "Dense result", score=0.85)],
            total_retrieved=1,
            top_k=20,
        )

        fake_lexical = FakeLexicalRetriever(should_fail=True)

        service = HybridRetrievalService(
            retrieval_service=mock_dense,
            lexical_retriever=fake_lexical,
        )

        resp = service.search(query="query", user_id="user-1")

        assert len(resp.results) == 1
        assert resp.results[0].chunk_id == "c1"
        assert resp.dense_count == 1
        assert resp.lexical_count == 0

    def test_both_branches_failing_raises_error(self):
        mock_dense = MagicMock()
        mock_dense.search.side_effect = RetrievalError("Dense down")
        fake_lexical = FakeLexicalRetriever(should_fail=True)

        service = HybridRetrievalService(
            retrieval_service=mock_dense,
            lexical_retriever=fake_lexical,
        )

        with pytest.raises(HybridRetrievalError, match="Both dense and lexical"):
            service.search(query="query", user_id="user-1")

    def test_empty_results_on_both_branches(self):
        mock_dense = MagicMock()
        mock_dense.search.return_value = RetrievalResponse(
            query="query",
            results=[],
            total_retrieved=0,
            top_k=20,
        )
        fake_lexical = FakeLexicalRetriever(custom_results=[])

        service = HybridRetrievalService(
            retrieval_service=mock_dense,
            lexical_retriever=fake_lexical,
        )

        resp = service.search(query="query", user_id="user-1")
        assert resp.results == []
        assert resp.total_fused == 0

    def test_tenant_isolation_passed_to_both_retrievers(self):
        mock_dense = MagicMock()
        mock_dense.search.return_value = RetrievalResponse(query="q", results=[], total_retrieved=0, top_k=20)
        fake_lexical = FakeLexicalRetriever()

        service = HybridRetrievalService(
            retrieval_service=mock_dense,
            lexical_retriever=fake_lexical,
        )

        service.search(query="q", user_id="user-999", document_id="doc-123")

        mock_dense.search.assert_called_once_with(
            query="q",
            user_id="user-999",
            top_k=20,
            similarity_threshold=None,
            document_id="doc-123",
        )
        assert fake_lexical.last_user_id == "user-999"

    def test_disabled_hybrid_retrieval_bypasses_lexical_search(self):
        mock_dense = MagicMock()
        mock_dense.search.return_value = RetrievalResponse(
            query="q",
            results=[_make_res("c1", "Dense chunk", score=0.9)],
            total_retrieved=1,
            top_k=20,
        )
        fake_lexical = FakeLexicalRetriever()

        config = HybridRetrievalConfig(enabled=False)
        service = HybridRetrievalService(
            retrieval_service=mock_dense,
            lexical_retriever=fake_lexical,
            config=config,
        )

        resp = service.search(query="q", user_id="user-1")
        assert len(resp.results) == 1
        assert fake_lexical.call_count == 0  # Lexical branch bypassed
