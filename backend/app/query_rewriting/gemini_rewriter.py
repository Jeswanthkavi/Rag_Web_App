"""Production Gemini-backed conversational query rewriter."""

import logging
from typing import Any, Dict, Optional
import httpx
from pydantic import SecretStr

from backend.app.core.config import get_settings
from backend.app.query_rewriting.base import (
    BaseQueryRewriter,
    QueryRewriterError,
    RewrittenQuery,
)
from backend.app.query_rewriting.config import QueryRewritingConfig
from backend.app.query_rewriting.prompts import (
    QUERY_REWRITING_SYSTEM_INSTRUCTION,
    build_rewriting_prompt,
)

logger = logging.getLogger("app.query_rewriting.gemini")

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiQueryRewriter(BaseQueryRewriter):
    """Query rewriter that calls the Google Gemini REST API."""

    def __init__(
        self,
        config: Optional[QueryRewritingConfig] = None,
        api_key: Optional[SecretStr] = None,
    ):
        settings = get_settings()
        self.config = config or QueryRewritingConfig.from_settings(settings)
        self._api_key = api_key if api_key is not None else settings.gemini.gemini_api_key

    @property
    def provider_name(self) -> str:
        return "gemini"

    def rewrite(
        self,
        query: str,
        conversation_history: Optional[str] = None,
    ) -> RewrittenQuery:
        """Invokes Gemini to rewrite the conversational query into a standalone retrieval query."""
        if not query or not query.strip():
            raise QueryRewriterError("Input query cannot be empty or whitespace-only.")

        cleaned_query = query.strip()

        # If no conversation history exists, return original query immediately
        if not conversation_history or not conversation_history.strip():
            return RewrittenQuery(
                original_query=cleaned_query,
                retrieval_query=cleaned_query,
                was_rewritten=False,
                reason="no_conversation_history",
            )

        key_value = (
            self._api_key.get_secret_value()
            if isinstance(self._api_key, SecretStr)
            else str(self._api_key or "")
        )
        if not key_value or not key_value.strip():
            raise QueryRewriterError("GEMINI_API_KEY is not configured for query rewriting.")

        url = f"{GEMINI_API_BASE_URL}/{self.config.model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": key_value,
        }

        user_prompt = build_rewriting_prompt(cleaned_query, conversation_history)
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": min(self.config.max_query_length // 2, 256),
            },
            "system_instruction": {
                "parts": [{"text": QUERY_REWRITING_SYSTEM_INSTRUCTION}]
            },
        }

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                sanitized_msg = response.text.replace(key_value, "[REDACTED_API_KEY]")
                logger.error(
                    f"Gemini query rewriting error HTTP {response.status_code}: {sanitized_msg[:200]}"
                )
                raise QueryRewriterError(
                    f"Gemini API error (HTTP {response.status_code}): {sanitized_msg[:200]}"
                )

            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                logger.warning("Gemini query rewriting returned 0 candidates.")
                return RewrittenQuery(
                    original_query=cleaned_query,
                    retrieval_query=cleaned_query,
                    was_rewritten=False,
                    reason="gemini_no_candidates",
                )

            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            if not parts or "text" not in parts[0]:
                logger.warning("Gemini query rewriting returned empty candidate parts.")
                return RewrittenQuery(
                    original_query=cleaned_query,
                    retrieval_query=cleaned_query,
                    was_rewritten=False,
                    reason="gemini_empty_parts",
                )

            rewritten_text = parts[0]["text"].strip()
            # Clean up any potential markdown code fence wrapping or quotes
            if rewritten_text.startswith("```") and rewritten_text.endswith("```"):
                lines = rewritten_text.splitlines()
                if len(lines) >= 3:
                    rewritten_text = "\n".join(lines[1:-1]).strip()
            if (rewritten_text.startswith('"') and rewritten_text.endswith('"')) or (
                rewritten_text.startswith("'") and rewritten_text.endswith("'")
            ):
                rewritten_text = rewritten_text[1:-1].strip()

            if not rewritten_text:
                return RewrittenQuery(
                    original_query=cleaned_query,
                    retrieval_query=cleaned_query,
                    was_rewritten=False,
                    reason="empty_output_fallback",
                )

            # Limit length if exceeded
            if len(rewritten_text) > self.config.max_query_length:
                rewritten_text = rewritten_text[: self.config.max_query_length].rstrip()

            was_changed = rewritten_text.lower() != cleaned_query.lower()
            return RewrittenQuery(
                original_query=cleaned_query,
                retrieval_query=rewritten_text,
                was_rewritten=was_changed,
                reason="gemini_rewritten" if was_changed else "standalone_unchanged",
            )

        except httpx.TimeoutException as e:
            logger.error(f"Gemini query rewriting timed out after {self.config.timeout}s")
            raise QueryRewriterError(f"Gemini query rewriting request timed out after {self.config.timeout}s") from e

        except httpx.RequestError as e:
            sanitized = str(e).replace(key_value, "[REDACTED_API_KEY]") if key_value else str(e)
            logger.error(f"Gemini query rewriting network error: {sanitized}")
            raise QueryRewriterError(f"Gemini query rewriting network error: {sanitized}") from e

        except QueryRewriterError:
            raise

        except Exception as e:
            sanitized = str(e).replace(key_value, "[REDACTED_API_KEY]") if key_value else str(e)
            logger.error(f"Unexpected error in Gemini query rewriter: {sanitized}")
            raise QueryRewriterError(f"Gemini query rewriting failed: {sanitized}") from e
