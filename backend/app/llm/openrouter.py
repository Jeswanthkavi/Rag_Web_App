"""OpenRouter LLM provider implementation using OpenAI-compatible Chat Completions REST API."""

import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from pydantic import SecretStr

from backend.app.core.config import OpenRouterSettings, get_settings
from backend.app.llm.base import BaseLLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger("app.llm.openrouter")


class OpenRouterProvider(BaseLLMProvider):
    """Production provider connecting to OpenRouter OpenAI-compatible REST API."""

    def __init__(
        self,
        settings: Optional[OpenRouterSettings] = None,
        api_key: Optional[SecretStr] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        app_name: Optional[str] = None,
        http_referer: Optional[str] = None,
    ):
        s = settings or get_settings().openrouter
        self._api_key = api_key if api_key is not None else s.openrouter_api_key
        self._model_name = model_name or s.openrouter_model
        self._base_url = (base_url or s.openrouter_base_url).rstrip("/")
        self._temperature = temperature if temperature is not None else s.openrouter_temperature
        self._max_tokens = max_tokens if max_tokens is not None else s.openrouter_max_tokens
        self._timeout = timeout if timeout is not None else s.openrouter_timeout
        self._app_name = app_name or s.openrouter_app_name
        self._http_referer = http_referer or s.openrouter_http_referer

    @property
    def provider_name(self) -> str:
        return "openrouter"

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
        """Executes a completion request against OpenRouter Chat Completions endpoint."""
        if not prompt or not prompt.strip():
            raise LLMProviderError("Prompt cannot be empty or whitespace-only.")

        key_value = (
            self._api_key.get_secret_value()
            if isinstance(self._api_key, SecretStr)
            else str(self._api_key or "")
        )
        if not key_value or not key_value.strip():
            raise LLMProviderError("OPENROUTER_API_KEY is not configured or is empty.")

        url = f"{self._base_url}/chat/completions"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_value}",
        }
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._app_name:
            headers["X-Title"] = self._app_name

        messages: List[Dict[str, str]] = []
        if system_instruction and system_instruction.strip():
            messages.append({"role": "system", "content": system_instruction.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                sanitized_msg = response.text.replace(key_value, "[REDACTED_API_KEY]")
                logger.error(
                    f"OpenRouter API returned error HTTP {response.status_code}: {sanitized_msg[:200]}"
                )

                if response.status_code == 401:
                    raise LLMProviderError("OpenRouter API error (HTTP 401): Authentication failed. Invalid or missing API key.")
                elif response.status_code == 403:
                    raise LLMProviderError("OpenRouter API error (HTTP 403): Access forbidden or key permissions insufficient.")
                elif response.status_code == 429:
                    raise LLMProviderError("OpenRouter provider is rate limited (HTTP 429). Please try again later.")
                elif response.status_code == 400:
                    raise LLMProviderError(f"OpenRouter API error (HTTP 400): Bad request - {sanitized_msg[:200]}")
                elif response.status_code == 408:
                    raise LLMProviderError("OpenRouter API request timed out (HTTP 408).")
                elif response.status_code >= 500:
                    raise LLMProviderError(f"OpenRouter API server error (HTTP {response.status_code}): {sanitized_msg[:200]}")
                else:
                    raise LLMProviderError(f"OpenRouter API error (HTTP {response.status_code}): {sanitized_msg[:200]}")

            try:
                data = response.json()
            except Exception as e:
                sanitized_body = response.text.replace(key_value, "[REDACTED_API_KEY]")
                logger.error(f"Failed to parse OpenRouter response as JSON: {sanitized_body[:200]}")
                raise LLMProviderError(f"OpenRouter returned malformed JSON: {sanitized_body[:200]}") from e

            choices = data.get("choices") or []
            if not choices:
                error_info = data.get("error")
                if error_info:
                    err_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
                    sanitized_err = err_msg.replace(key_value, "[REDACTED_API_KEY]")
                    logger.error(f"OpenRouter returned error in body: {sanitized_err[:200]}")
                    raise LLMProviderError(f"OpenRouter API error: {sanitized_err[:200]}")
                logger.warning("OpenRouter returned 0 choices.")
                raise LLMProviderError("OpenRouter generation returned no choices")

            first_choice = choices[0]
            message = first_choice.get("message") or {}
            content = message.get("content")

            if content is None:
                logger.warning("OpenRouter choice contains no message content.")
                raise LLMProviderError("OpenRouter choice contains no message content")

            finish_reason = first_choice.get("finish_reason")
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")

            return LLMResponse(
                text=content,
                model_name=self._model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )

        except httpx.TimeoutException as e:
            logger.error(f"OpenRouter API request timed out after {self._timeout}s")
            raise LLMProviderError(f"OpenRouter request timed out after {self._timeout}s") from e

        except httpx.RequestError as e:
            sanitized = str(e).replace(key_value, "[REDACTED_API_KEY]") if key_value else str(e)
            logger.error(f"OpenRouter network request failed: {sanitized}")
            raise LLMProviderError(f"OpenRouter network error: {sanitized}") from e

        except LLMProviderError:
            raise

        except Exception as e:
            sanitized = str(e).replace(key_value, "[REDACTED_API_KEY]") if key_value else str(e)
            logger.error(f"Unexpected error during OpenRouter generation: {sanitized}")
            raise LLMProviderError(f"OpenRouter generation failed: {sanitized}") from e
