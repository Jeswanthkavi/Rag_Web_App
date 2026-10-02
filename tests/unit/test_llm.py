"""Unit tests for LLM provider abstraction, GeminiProvider REST client, and FakeLLMProvider."""

import pytest
from unittest.mock import MagicMock, patch
import httpx
from pydantic import SecretStr

from backend.app.core.config import GeminiSettings
from backend.app.llm.base import LLMProviderError, LLMResponse
from backend.app.llm.fake_provider import FakeLLMProvider
from backend.app.llm.gemini import GeminiProvider


class TestFakeLLMProvider:
    """Unit tests for deterministic FakeLLMProvider."""

    def test_fake_provider_defaults(self):
        provider = FakeLLMProvider(model_name="test-model")
        assert provider.provider_name == "fake"
        assert provider.model_name == "test-model"

        resp = provider.generate(prompt="What is RAG?", system_instruction="Be concise")
        assert "Grounded answer" in resp.text
        assert resp.model_name == "test-model"
        assert resp.finish_reason == "STOP"
        assert provider.last_prompt == "What is RAG?"
        assert provider.last_system_instruction == "Be concise"

    def test_fake_provider_custom_response(self):
        provider = FakeLLMProvider(custom_response="Fixed answer text.")
        resp = provider.generate(prompt="Any question")
        assert resp.text == "Fixed answer text."


class TestGeminiProviderUnit:
    """Unit tests for GeminiProvider HTTP interactions with mocked httpx."""

    def test_empty_prompt_raises_error(self):
        provider = GeminiProvider(api_key=SecretStr("fake-key"))
        with pytest.raises(LLMProviderError, match="Prompt cannot be empty"):
            provider.generate(prompt="")

        with pytest.raises(LLMProviderError, match="Prompt cannot be empty"):
            provider.generate(prompt="   ")

    def test_missing_api_key_raises_error(self):
        provider = GeminiProvider(api_key=SecretStr(""))
        with pytest.raises(LLMProviderError, match="GEMINI_API_KEY is not configured"):
            provider.generate(prompt="Valid prompt")

    def test_gemini_successful_generation(self):
        provider = GeminiProvider(
            api_key=SecretStr("test-secret-api-key"),
            model_name="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=1000,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini generated answer."}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 25,
                "candidatesTokenCount": 10,
            },
        }

        with patch("httpx.Client.post", return_value=mock_response) as mock_post:
            resp: LLMResponse = provider.generate(
                prompt="User question with context",
                system_instruction="Strict grounding rules",
            )

            assert resp.text == "Gemini generated answer."
            assert resp.model_name == "gemini-2.5-flash"
            assert resp.prompt_tokens == 25
            assert resp.completion_tokens == 10
            assert resp.finish_reason == "STOP"

            # Verify request payload and headers
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            headers = call_kwargs["headers"]
            assert headers["x-goog-api-key"] == "test-secret-api-key"
            payload = call_kwargs["json"]
            assert payload["contents"][0]["parts"][0]["text"] == "User question with context"
            assert payload["system_instruction"]["parts"][0]["text"] == "Strict grounding rules"
            assert payload["generationConfig"]["temperature"] == 0.3
            assert payload["generationConfig"]["maxOutputTokens"] == 1000

    def test_gemini_http_error_redacts_api_key(self):
        secret_key = "very-secret-gemini-key-12345"
        provider = GeminiProvider(
            api_key=SecretStr(secret_key),
            model_name="gemini-2.5-flash",
        )

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = f"Permission denied for key {secret_key}: quota exceeded"

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError) as exc_info:
                provider.generate(prompt="Sample question")

            err_str = str(exc_info.value)
            assert secret_key not in err_str
            assert "[REDACTED_API_KEY]" in err_str

    def test_gemini_empty_candidates_safety_block_handled(self):
        provider = GeminiProvider(api_key=SecretStr("test-key"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [],
            "promptFeedback": {"blockReason": "SAFETY"},
        }

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="returned no candidates.*SAFETY"):
                provider.generate(prompt="Sample question")

    def test_gemini_candidate_missing_text_parts(self):
        provider = GeminiProvider(api_key=SecretStr("test-key"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": []},
                    "finishReason": "RECITATION",
                }
            ]
        }

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="candidate empty.*RECITATION"):
                provider.generate(prompt="Sample question")

    def test_gemini_timeout_handled_cleanly(self):
        provider = GeminiProvider(api_key=SecretStr("test-key"), timeout=5.0)

        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timeout")):
            with pytest.raises(LLMProviderError, match="timed out after 5.0s"):
                provider.generate(prompt="Sample question")

    def test_gemini_network_error_handled_cleanly(self):
        provider = GeminiProvider(api_key=SecretStr("test-key"))

        with patch("httpx.Client.post", side_effect=httpx.ConnectError("DNS resolution failed")):
            with pytest.raises(LLMProviderError, match="Gemini network error"):
                provider.generate(prompt="Sample question")
