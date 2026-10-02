"""Unit tests for LLM provider abstraction, GeminiProvider REST client, and FakeLLMProvider."""

import pytest
from unittest.mock import MagicMock, patch
import httpx
from pydantic import SecretStr

from backend.app.core.config import GeminiSettings, OpenRouterSettings
from backend.app.llm.base import LLMProviderError, LLMResponse
from backend.app.llm.fake_provider import FakeLLMProvider
from backend.app.llm.gemini import GeminiProvider
from backend.app.llm.openrouter import OpenRouterProvider


class TestOpenRouterProviderUnit:
    """Unit tests for OpenRouterProvider HTTP interactions with mocked httpx."""

    def test_empty_prompt_raises_error(self):
        provider = OpenRouterProvider(api_key=SecretStr("fake-key"))
        with pytest.raises(LLMProviderError, match="Prompt cannot be empty"):
            provider.generate(prompt="")

        with pytest.raises(LLMProviderError, match="Prompt cannot be empty"):
            provider.generate(prompt="   ")

    def test_missing_api_key_raises_error(self):
        provider = OpenRouterProvider(api_key=SecretStr(""))
        with pytest.raises(LLMProviderError, match="OPENROUTER_API_KEY is not configured"):
            provider.generate(prompt="Valid prompt")

    def test_openrouter_successful_generation(self):
        provider = OpenRouterProvider(
            api_key=SecretStr("test-openrouter-secret-key"),
            model_name="nvidia/nemotron-3-ultra-550b-a55b:free",
            temperature=0.3,
            max_tokens=1000,
            app_name="Advanced-RAG-Platform",
            http_referer="http://localhost:8000",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Nemotron generated grounded answer.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 15,
            },
        }

        with patch("httpx.Client.post", return_value=mock_response) as mock_post:
            resp: LLMResponse = provider.generate(
                prompt="User question with context",
                system_instruction="Strict grounding rules",
            )

            assert resp.text == "Nemotron generated grounded answer."
            assert resp.model_name == "nvidia/nemotron-3-ultra-550b-a55b:free"
            assert resp.prompt_tokens == 30
            assert resp.completion_tokens == 15
            assert resp.finish_reason == "stop"

            # Verify request payload, headers, and endpoint
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            assert call_url == "https://openrouter.ai/api/v1/chat/completions"

            call_kwargs = mock_post.call_args[1]
            headers = call_kwargs["headers"]
            assert headers["Authorization"] == "Bearer test-openrouter-secret-key"
            assert headers["Content-Type"] == "application/json"
            assert headers["HTTP-Referer"] == "http://localhost:8000"
            assert headers["X-Title"] == "Advanced-RAG-Platform"

            payload = call_kwargs["json"]
            assert payload["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
            assert payload["temperature"] == 0.3
            assert payload["max_tokens"] == 1000
            assert payload["messages"] == [
                {"role": "system", "content": "Strict grounding rules"},
                {"role": "user", "content": "User question with context"},
            ]

    def test_openrouter_http_401_handled(self):
        provider = OpenRouterProvider(api_key=SecretStr("bad-key"))
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = '{"error": {"message": "Invalid API Key"}}'

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="Authentication failed"):
                provider.generate(prompt="Sample question")

    def test_openrouter_http_429_rate_limit_handled(self):
        provider = OpenRouterProvider(api_key=SecretStr("valid-key"))
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = '{"error": {"message": "Rate limit exceeded"}}'

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="rate limited"):
                provider.generate(prompt="Sample question")

    def test_openrouter_http_500_handled(self):
        provider = OpenRouterProvider(api_key=SecretStr("valid-key"))
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = '{"error": {"message": "Bad Gateway"}}'

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="server error"):
                provider.generate(prompt="Sample question")

    def test_openrouter_secret_redaction_on_error(self):
        secret_key = "sk-or-v1-my-secret-token-9999"
        provider = OpenRouterProvider(api_key=SecretStr(secret_key))
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = f"Invalid request with token {secret_key}"

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError) as exc_info:
                provider.generate(prompt="Sample question")
            err_str = str(exc_info.value)
            assert secret_key not in err_str
            assert "[REDACTED_API_KEY]" in err_str

    def test_openrouter_empty_choices_handled(self):
        provider = OpenRouterProvider(api_key=SecretStr("valid-key"))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="returned no choices"):
                provider.generate(prompt="Sample question")

    def test_openrouter_missing_content_handled(self):
        provider = OpenRouterProvider(api_key=SecretStr("valid-key"))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {}, "finish_reason": "length"}]
        }

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="no message content"):
                provider.generate(prompt="Sample question")

    def test_openrouter_malformed_json_handled(self):
        provider = OpenRouterProvider(api_key=SecretStr("valid-key"))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Malformed JSON")
        mock_response.text = "Not JSON"

        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="malformed JSON"):
                provider.generate(prompt="Sample question")

    def test_openrouter_timeout_handled(self):
        provider = OpenRouterProvider(api_key=SecretStr("valid-key"), timeout=5.0)

        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timeout")):
            with pytest.raises(LLMProviderError, match="timed out after 5.0s"):
                provider.generate(prompt="Sample question")

    def test_openrouter_network_error_handled(self):
        provider = OpenRouterProvider(api_key=SecretStr("valid-key"))

        with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(LLMProviderError, match="OpenRouter network error"):
                provider.generate(prompt="Sample question")


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
