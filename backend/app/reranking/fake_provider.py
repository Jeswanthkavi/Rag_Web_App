"""Deterministic mock reranker for testing and offline development."""

from typing import Callable, Dict, List, Optional
from backend.app.reranking.base import BaseReranker, RerankerError
from backend.app.reranking.models import RerankResult
from backend.app.retrieval.models import RetrievalResult


class FakeReranker(BaseReranker):
    """Deterministic in-memory reranker for unit and integration testing."""

    def __init__(
        self,
        custom_scores: Optional[Dict[str, float]] = None,
        scoring_fn: Optional[Callable[[str, str], float]] = None,
        should_fail: bool = False,
        reverse_order: bool = False,
    ):
        self._custom_scores = custom_scores or {}
        self._scoring_fn = scoring_fn
        self._should_fail = should_fail
        self._reverse_order = reverse_order
        self.call_count = 0
        self.last_query: Optional[str] = None
        self.last_candidates_count: int = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def set_custom_score(self, chunk_id: str, score: float) -> None:
        self._custom_scores[chunk_id] = score

    def set_should_fail(self, should_fail: bool) -> None:
        self._should_fail = should_fail

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Deterministically scores and reorders candidates."""
        self.call_count += 1
        self.last_query = query
        self.last_candidates_count = len(candidates)

        if self._should_fail:
            raise RerankerError("Simulated reranker provider failure.")

        if not candidates:
            return []

        if not query or not query.strip():
            raise RerankerError("Query cannot be empty for reranking.")

        cleaned_query = query.strip().lower()
        query_terms = set(re_split for re_split in cleaned_query.split() if len(re_split) > 1)

        scored: List[tuple[float, RetrievalResult]] = []
        for idx, cand in enumerate(candidates):
            # Check explicit custom score for this chunk_id
            if cand.chunk_id in self._custom_scores:
                score = self._custom_scores[cand.chunk_id]
            elif self._scoring_fn:
                score = self._scoring_fn(cleaned_query, cand.chunk_text)
            elif self._reverse_order:
                # Reverse rank for testing reordering (higher idx gets higher score)
                score = float(idx + 1)
            else:
                # Deterministic term overlap scoring
                chunk_terms = set(cand.chunk_text.lower().split())
                overlap = len(query_terms.intersection(chunk_terms))
                # Add fractional tie-breaker based on original dense score
                score = float(overlap * 2.0) + (cand.score * 0.1)

            scored.append((score, cand))

        # Sort descending by rerank score
        scored.sort(key=lambda x: x[0], reverse=True)

        limit = top_k if top_k is not None else len(scored)
        sliced = scored[:limit]

        results: List[RerankResult] = []
        for rank_idx, (rerank_score, cand) in enumerate(sliced, start=1):
            results.append(
                RerankResult.from_retrieval_result(
                    result=cand,
                    rerank_score=rerank_score,
                    rank=rank_idx,
                )
            )

        return results
