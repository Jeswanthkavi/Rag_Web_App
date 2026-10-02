"""Query rewriting service coordinating rewriter providers, fallback logic, and security safeguards."""

import logging
from typing import Optional

from backend.app.query_rewriting.base import (
    BaseQueryRewriter,
    QueryRewriterError,
    RewrittenQuery,
)
from backend.app.query_rewriting.config import QueryRewritingConfig
from backend.app.query_rewriting.openrouter_rewriter import OpenRouterQueryRewriter

logger = logging.getLogger("app.query_rewriting.service")


class QueryRewritingService:
    """Service orchestrating conversational query rewriting with safe fallback mechanisms."""

    def __init__(
        self,
        rewriter: Optional[BaseQueryRewriter] = None,
        config: Optional[QueryRewritingConfig] = None,
    ):
        self.config = config or QueryRewritingConfig.from_settings()
        self.rewriter = rewriter or OpenRouterQueryRewriter(config=self.config)

    def rewrite_query(
        self,
        query: str,
        conversation_history: Optional[str] = None,
    ) -> RewrittenQuery:
        """Rewrites a conversational user query into a standalone retrieval query.
        
        Guarantees:
        1. Standalone Bypass: If query rewriting is disabled, or no conversation history exists,
           returns the original query immediately without invoking the rewriter.
        2. Safe Fallback: If provider invocation fails or raises an error, gracefully falls back
           to returning the original query (retrieval_query = original_query) without failing the request.
        3. Length / Empty Guard: If the rewritten query is empty or whitespace, falls back to original_query.
        """
        if query is None:
            raise QueryRewriterError("Query cannot be None.")

        sanitized_query = query.strip()
        if not sanitized_query:
            return RewrittenQuery(
                original_query="",
                retrieval_query="",
                was_rewritten=False,
                reason="empty_query",
            )

        # 1. Standalone bypass checks
        if not self.config.enabled:
            return RewrittenQuery(
                original_query=sanitized_query,
                retrieval_query=sanitized_query,
                was_rewritten=False,
                reason="rewriting_disabled",
            )

        if not conversation_history or not conversation_history.strip():
            return RewrittenQuery(
                original_query=sanitized_query,
                retrieval_query=sanitized_query,
                was_rewritten=False,
                reason="no_history_bypass",
            )

        # 2. Invoke provider with fallback protection
        try:
            result = self.rewriter.rewrite(
                query=sanitized_query,
                conversation_history=conversation_history,
            )

            # Ensure valid retrieval query returned
            retrieval_q = result.retrieval_query.strip() if result.retrieval_query else ""
            if not retrieval_q:
                logger.warning("Query rewriter returned empty text; falling back to original query.")
                return RewrittenQuery(
                    original_query=sanitized_query,
                    retrieval_query=sanitized_query,
                    was_rewritten=False,
                    reason="empty_output_fallback",
                )

            # Enforce max length
            if len(retrieval_q) > self.config.max_query_length:
                retrieval_q = retrieval_q[: self.config.max_query_length].rstrip()

            return RewrittenQuery(
                original_query=sanitized_query,
                retrieval_query=retrieval_q,
                was_rewritten=result.was_rewritten and (retrieval_q.lower() != sanitized_query.lower()),
                reason=result.reason or "rewritten_success",
            )

        except Exception as e:
            # Safe diagnostic logging without sensitive conversation history or secrets
            err_type = type(e).__name__
            logger.warning(
                f"Query rewriting provider failed ({err_type}). Falling back to original query for retrieval."
            )
            return RewrittenQuery(
                original_query=sanitized_query,
                retrieval_query=sanitized_query,
                was_rewritten=False,
                reason=f"fallback_on_error_{err_type}",
            )
