"""Production OpenRouter-backed conversational query rewriter."""

import logging
from typing import Any, Dict, List, Optional
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

logger = logging.getLogger("app.query_rewriting.openrouter")


class OpenRouterQueryRewriter(BaseQueryRewriter):
    """Query rewriter that calls the OpenRouter OpenAI-compatible REST API."""

    def __init__(
        self,
        config: Optional[QueryRewritingConfig] = None,
        api_key: Optional[SecretStr] = None,
        base_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.config = config or QueryRewritingConfig.from_settings(settings)
        self._api_key = api_key if api_key is not None else settings.openrouter.openrouter_api_key
        self._base_url = (base_url or settings.openrouter.openrouter_base_url).rstrip("/")
        self._app_name = settings.openrouter.openrouter_app_name
        self._http_referer = settings.openrouter.openrouter_http_referer

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def rewrite(
        self,
        query: str,
        conversation_history: Optional[str] = None,
    ) -> RewrittenQuery:
        """Invokes OpenRouter to rewrite the conversational query into a standalone retrieval query."""
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
            raise QueryRewriterError("OPENROUTER_API_KEY is not configured for query rewriting.")

        url = f"{self._base_url}/chat/completions"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_value}",
        }
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._app_name:
            headers["X-Title"] = self._app_name

        user_prompt = build_rewriting_prompt(cleaned_query, conversation_history)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": QUERY_REWRITING_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ]

        payload: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": min(self.config.max_query_length // 2, 256),
        }

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                sanitized_msg = response.text.replace(key_value, "[REDACTED_API_KEY]")
                logger.error(
                    f"OpenRouter query rewriting error HTTP {response.status_code}: {sanitized_msg[:200]}"
                )
                if response.status_code == 429:
                    raise QueryRewriterError("OpenRouter query rewriting provider is rate limited (HTTP 429).")
                raise QueryRewriterError(
                    f"OpenRouter API error (HTTP {response.status_code}): {sanitized_msg[:200]}"
                )

            try:
                data = response.json()
            except Exception as e:
                sanitized_body = response.text.replace(key_value, "[REDACTED_API_KEY]")
                logger.error(f"Failed to parse OpenRouter query rewriting response as JSON: {sanitized_body[:200]}")
                raise QueryRewriterError(f"OpenRouter returned malformed JSON: {sanitized_body[:200]}") from e

            choices = data.get("choices") or []
            if not choices:
                error_info = data.get("error")
                if error_info:
                    err_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
                    sanitized_err = err_msg.replace(key_value, "[REDACTED_API_KEY]")
                    logger.error(f"OpenRouter query rewriting returned error: {sanitized_err[:200]}")
                    raise QueryRewriterError(f"OpenRouter query rewriting error: {sanitized_err[:200]}")
                logger.warning("OpenRouter query rewriting returned 0 choices.")
                return RewrittenQuery(
                    original_query=cleaned_query,
                    retrieval_query=cleaned_query,
                    was_rewritten=False,
                    reason="openrouter_no_choices",
                )

            message = choices[0].get("message") or {}
            rewritten_text = message.get("content")
            if rewritten_text is None or not isinstance(rewritten_text, str):
                logger.warning("OpenRouter query rewriting returned empty or non-string message content.")
                return RewrittenQuery(
                    original_query=cleaned_query,
                    retrieval_query=cleaned_query,
                    was_rewritten=False,
                    reason="openrouter_empty_content",
                )

            rewritten_text = rewritten_text.strip()
            # Clean up potential markdown code fence wrapping or quotes
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
                reason="openrouter_rewritten" if was_changed else "standalone_unchanged",
            )

        except httpx.TimeoutException as e:
            logger.error(f"OpenRouter query rewriting timed out after {self.config.timeout}s")
            raise QueryRewriterError(f"OpenRouter query rewriting request timed out after {self.config.timeout}s") from e

        except httpx.RequestError as e:
            sanitized = str(e).replace(key_value, "[REDACTED_API_KEY]") if key_value else str(e)
            logger.error(f"OpenRouter query rewriting network error: {sanitized}")
            raise QueryRewriterError(f"OpenRouter query rewriting network error: {sanitized}") from e

        except QueryRewriterError:
            raise

        except Exception as e:
            sanitized = str(e).replace(key_value, "[REDACTED_API_KEY]") if key_value else str(e)
            logger.error(f"Unexpected error in OpenRouter query rewriter: {sanitized}")
            raise QueryRewriterError(f"OpenRouter query rewriting failed: {sanitized}") from e
