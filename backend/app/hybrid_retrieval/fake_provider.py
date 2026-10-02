"""Deterministic mock lexical retriever for testing and offline development."""

from typing import Dict, List, Optional
from backend.app.hybrid_retrieval.base import BaseLexicalRetriever, LexicalRetrievalError
from backend.app.retrieval.models import RetrievalResult


class FakeLexicalRetriever(BaseLexicalRetriever):
    """Deterministic in-memory lexical retriever for unit and integration tests."""

    def __init__(
        self,
        custom_results: Optional[List[RetrievalResult]] = None,
        chunk_keyword_map: Optional[Dict[str, List[RetrievalResult]]] = None,
        should_fail: bool = False,
    ):
        self._custom_results = custom_results or []
        self._chunk_keyword_map = chunk_keyword_map or {}
        self._should_fail = should_fail
        self.call_count = 0
        self.last_query: Optional[str] = None
        self.last_user_id: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "fake_lexical"

    def set_should_fail(self, should_fail: bool) -> None:
        self._should_fail = should_fail

    def set_custom_results(self, results: List[RetrievalResult]) -> None:
        self._custom_results = results

    def search(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Returns deterministic lexical results for testing."""
        self.call_count += 1
        self.last_query = query
        self.last_user_id = user_id

        if self._should_fail:
            raise LexicalRetrievalError("Simulated lexical retrieval provider failure.")

        if not user_id or not user_id.strip():
            raise LexicalRetrievalError("user_id is required.")

        if not query or not query.strip():
            return []

        # Check keyword map
        cleaned_query = query.strip().lower()
        results: List[RetrievalResult] = []
        for kw, res_list in self._chunk_keyword_map.items():
            if kw.lower() in cleaned_query:
                results.extend(res_list)

        if not results:
            results = list(self._custom_results)

        # Apply tenant filter
        filtered = [r for r in results if r.user_id == user_id]
        if document_id:
            filtered = [r for r in filtered if r.document_id == document_id]

        limit = top_k if top_k is not None else len(filtered)
        sliced = filtered[:limit]

        # Re-assign 1-indexed rank
        ranked = []
        for idx, r in enumerate(sliced, start=1):
            ranked.append(
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    user_id=r.user_id,
                    chunk_index=r.chunk_index,
                    chunk_text=r.chunk_text,
                    page_number=r.page_number,
                    filename=r.filename,
                    file_type=r.file_type,
                    token_count=r.token_count,
                    score=r.score,
                    rank=idx,
                )
            )
        return ranked
