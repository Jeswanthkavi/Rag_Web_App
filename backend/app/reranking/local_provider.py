"""Local cross-encoder reranker provider using sentence-transformers."""

import logging
from typing import List, Optional
from backend.app.reranking.base import BaseReranker, RerankerError
from backend.app.reranking.config import RerankingConfig
from backend.app.reranking.models import RerankResult
from backend.app.retrieval.models import RetrievalResult

logger = logging.getLogger("app.reranking.local")


class LocalReranker(BaseReranker):
    """Local inference reranker using CrossEncoder."""

    def __init__(self, config: Optional[RerankingConfig] = None):
        self.config = config or RerankingConfig.from_settings()
        self._model = None

    @property
    def provider_name(self) -> str:
        return "local"

    def _get_model(self):
        """Lazily loads and caches the CrossEncoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading local cross-encoder model: {self.config.model_name}")
                self._model = CrossEncoder(self.config.model_name)
            except Exception as e:
                sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
                logger.error(f"Failed to load cross-encoder model '{self.config.model_name}': {sanitized}")
                raise RerankerError(f"Failed to load reranker model '{self.config.model_name}': {sanitized}") from e
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Reranks candidate chunks by cross-encoder scoring against the query."""
        if not candidates:
            return []

        if not query or not query.strip():
            raise RerankerError("Query cannot be empty for reranking.")

        cleaned_query = query.strip()
        model = self._get_model()

        try:
            # Prepare pairs of (query, chunk_text)
            pairs = [[cleaned_query, cand.chunk_text] for cand in candidates]

            # Predict cross-encoder scores in batch
            raw_scores = model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            # Build intermediate candidate list with rerank scores
            scored_candidates = []
            for cand, score in zip(candidates, raw_scores):
                scored_candidates.append((float(score), cand))

            # Sort descending by rerank score
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # Limit to top_k
            limit = top_k if top_k is not None else self.config.final_top_k
            sliced = scored_candidates[:limit]

            # Build final RerankResult list with 1-indexed rank
            reranked_results: List[RerankResult] = []
            for rank_idx, (rerank_score, cand) in enumerate(sliced, start=1):
                reranked_results.append(
                    RerankResult.from_retrieval_result(
                        result=cand,
                        rerank_score=rerank_score,
                        rank=rank_idx,
                    )
                )

            return reranked_results

        except RerankerError:
            raise
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Cross-encoder reranking execution failed: {sanitized}")
            raise RerankerError(f"Reranking execution failed: {sanitized}") from e
