"""Deterministic mock query rewriter for testing and offline development."""

import re
from typing import Dict, Optional
from backend.app.query_rewriting.base import (
    BaseQueryRewriter,
    QueryRewriterError,
    RewrittenQuery,
)


class FakeQueryRewriter(BaseQueryRewriter):
    """Deterministic in-memory query rewriter for unit and integration testing."""

    def __init__(
        self,
        custom_mappings: Optional[Dict[str, str]] = None,
        should_fail: bool = False,
        return_empty: bool = False,
        max_query_length: int = 500,
    ):
        self._custom_mappings: Dict[str, str] = custom_mappings or {}
        self._should_fail = should_fail
        self._return_empty = return_empty
        self._max_query_length = max_query_length
        self.call_count = 0
        self.last_query: Optional[str] = None
        self.last_history: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "fake"

    def set_mapping(self, query: str, rewritten: str) -> None:
        """Adds or updates a deterministic mapping for a specific query."""
        self._custom_mappings[query.strip().lower()] = rewritten.strip()

    def set_should_fail(self, should_fail: bool) -> None:
        self._should_fail = should_fail

    def rewrite(
        self,
        query: str,
        conversation_history: Optional[str] = None,
    ) -> RewrittenQuery:
        """Deterministically rewrites conversational queries based on mappings or heuristics."""
        self.call_count += 1
        self.last_query = query
        self.last_history = conversation_history

        if self._should_fail:
            raise QueryRewriterError("Simulated query rewriter provider failure.")

        if not query or not query.strip():
            raise QueryRewriterError("Input query cannot be empty or whitespace-only.")

        cleaned_query = query.strip()

        # Standalone check: if no history or empty history, return original
        if not conversation_history or not conversation_history.strip():
            return RewrittenQuery(
                original_query=cleaned_query,
                retrieval_query=cleaned_query,
                was_rewritten=False,
                reason="no_conversation_history",
            )

        if self._return_empty:
            return RewrittenQuery(
                original_query=cleaned_query,
                retrieval_query="",
                was_rewritten=True,
                reason="empty_simulated",
            )

        # Check explicit custom mappings (case-insensitive)
        lower_q = cleaned_query.lower()
        if lower_q in self._custom_mappings:
            rewritten = self._custom_mappings[lower_q]
            if len(rewritten) > self._max_query_length:
                rewritten = rewritten[: self._max_query_length]
            return RewrittenQuery(
                original_query=cleaned_query,
                retrieval_query=rewritten,
                was_rewritten=rewritten.lower() != lower_q,
                reason="custom_mapping",
            )

        # Check partial mapping matches
        for key, val in self._custom_mappings.items():
            if key in lower_q:
                rewritten = lower_q.replace(key, val)
                if len(rewritten) > self._max_query_length:
                    rewritten = rewritten[: self._max_query_length]
                return RewrittenQuery(
                    original_query=cleaned_query,
                    retrieval_query=rewritten,
                    was_rewritten=True,
                    reason="partial_mapping",
                )

        # Simple heuristic pronoun / follow-up resolution based on conversation history
        # Look for the last topic mentioned in user or assistant history
        history_lines = [line.strip() for line in conversation_history.splitlines() if line.strip()]
        last_topic = ""
        for line in reversed(history_lines):
            # Extract possible topic
            if "warranty" in line.lower():
                last_topic = "warranty coverage"
                break
            elif "battery" in line.lower():
                last_topic = "battery"
                break
            elif "pricing" in line.lower() or "cost" in line.lower():
                last_topic = "pricing"
                break

        # Check if query has pronouns or conversational follow-up triggers
        follow_up_patterns = [
            (r"^what about (?:the )?(.+)\??$", r"What is the {topic} for \1?"),
            (r"^how about (?:the )?(.+)\??$", r"What is the {topic} for \1?"),
            (r"\bit\b", "{topic}"),
            (r"\bthis\b", "{topic}"),
            (r"\bthat\b", "{topic}"),
            (r"\bthey\b", "{topic}"),
        ]

        rewritten = cleaned_query
        was_rewritten = False

        if last_topic:
            for pattern, replacement in follow_up_patterns:
                if re.search(pattern, cleaned_query, re.IGNORECASE):
                    rep = replacement.replace("{topic}", last_topic)
                    rewritten = re.sub(pattern, rep, cleaned_query, flags=re.IGNORECASE).strip()
                    was_rewritten = True
                    break

        if len(rewritten) > self._max_query_length:
            rewritten = rewritten[: self._max_query_length]

        return RewrittenQuery(
            original_query=cleaned_query,
            retrieval_query=rewritten,
            was_rewritten=was_rewritten,
            reason="heuristic_resolution" if was_rewritten else "standalone_unchanged",
        )
