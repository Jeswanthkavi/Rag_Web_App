"""Deterministic in-memory Fake LLM provider for unit and integration testing."""

from typing import Optional
from backend.app.llm.base import BaseLLMProvider, LLMResponse


class FakeLLMProvider(BaseLLMProvider):
    """Fake LLM provider returning deterministic grounded completions for testing."""

    def __init__(
        self,
        model_name: str = "fake-gemini-model",
        custom_response: Optional[str] = None,
    ):
        self._model_name = model_name
        self._custom_response = custom_response
        self.call_count: int = 0
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "fake"

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
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction

        if self._custom_response is not None:
            text = self._custom_response
        else:
            text = f"Grounded answer generated from context for query in prompt (model={self._model_name})."

        return LLMResponse(
            text=text,
            model_name=self._model_name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            finish_reason="STOP",
        )
