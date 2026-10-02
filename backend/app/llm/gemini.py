"""Google Gemini LLM provider implementation using standard HTTP REST API."""

import logging
from typing import Any, Dict, Optional
import httpx
from pydantic import SecretStr

from backend.app.core.config import GeminiSettings, get_settings
from backend.app.llm.base import BaseLLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger("app.llm.gemini")

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseLLMProvider):
    """Production provider connecting to Google Gemini REST API."""

    def __init__(
        self,
        settings: Optional[GeminiSettings] = None,
        api_key: Optional[SecretStr] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        s = settings or get_settings().gemini
        self._api_key = api_key if api_key is not None else s.gemini_api_key
        self._model_name = model_name or s.gemini_model
        self._temperature = temperature if temperature is not None else s.gemini_temperature
        self._max_tokens = max_tokens if max_tokens is not None else s.gemini_max_tokens
        self._timeout = timeout if timeout is not None else s.gemini_timeout

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Executes a completion request against Google Gemini."""
        if not prompt or not prompt.strip():
            raise LLMProviderError("Prompt cannot be empty or whitespace-only.")

        key_value = self._api_key.get_secret_value() if isinstance(self._api_key, SecretStr) else str(self._api_key)
        if not key_value or not key_value.strip():
            raise LLMProviderError("GEMINI_API_KEY is not configured or is empty.")

        url = f"{GEMINI_API_BASE_URL}/{self._model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": key_value,
        }

        # Construct request payload
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature if temperature is not None else self._temperature,
                "maxOutputTokens": max_tokens if max_tokens is not None else self._max_tokens,
            },
        }

        if system_instruction and system_instruction.strip():
            payload["system_instruction"] = {
                "parts": [{"text": system_instruction.strip()}]
            }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                # Sanitize error detail to prevent key/secret leaks
                sanitized_msg = response.text.replace(key_value, "[REDACTED_API_KEY]")
                logger.error(
                    f"Gemini API returned error HTTP {response.status_code}: {sanitized_msg[:200]}"
                )
                raise LLMProviderError(
                    f"Gemini API error (HTTP {response.status_code}): {sanitized_msg[:200]}"
                )

            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                # Check for block reasons (safety/recitation)
                prompt_feedback = data.get("promptFeedback") or {}
                block_reason = prompt_feedback.get("blockReason", "Unknown")
                logger.warning(f"Gemini returned 0 candidates. Block reason: {block_reason}")
                raise LLMProviderError(f"Gemini generation returned no candidates (Block reason: {block_reason})")

            first_candidate = candidates[0]
            content = first_candidate.get("content") or {}
            parts = content.get("parts") or []
            if not parts or "text" not in parts[0]:
                finish_reason = first_candidate.get("finishReason", "UNKNOWN")
                logger.warning(f"Gemini candidate contains no text parts. Finish reason: {finish_reason}")
                raise LLMProviderError(f"Gemini candidate empty (Finish reason: {finish_reason})")

            generated_text = parts[0]["text"]
            finish_reason = first_candidate.get("finishReason")

            # Extract usage metadata if present
            usage = data.get("usageMetadata") or {}
            prompt_tokens = usage.get("promptTokenCount")
            completion_tokens = usage.get("candidatesTokenCount")

            return LLMResponse(
                text=generated_text,
                model_name=self._model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )

        except httpx.TimeoutException as e:
            logger.error(f"Gemini API request timed out after {self._timeout}s")
            raise LLMProviderError(f"Gemini request timed out after {self._timeout}s") from e

        except httpx.RequestError as e:
            sanitized = str(e).replace(key_value, "[REDACTED_API_KEY]") if key_value else str(e)
            logger.error(f"Gemini network request failed: {sanitized}")
            raise LLMProviderError(f"Gemini network error: {sanitized}") from e

        except LLMProviderError:
            raise

        except Exception as e:
            sanitized = str(e).replace(key_value, "[REDACTED_API_KEY]") if key_value else str(e)
            logger.error(f"Unexpected error during Gemini generation: {sanitized}")
            raise LLMProviderError(f"Gemini generation failed: {sanitized}") from e
