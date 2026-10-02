"""Abstract base classes and response models for LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class LLMProviderError(Exception):
    """Base exception for LLM provider operations and API errors."""
    pass


class LLMResponse(BaseModel):
    """Structured response returned by an LLM provider."""
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., description="The generated completion text")
    model_name: str = Field(..., description="Model identifier used for generation")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of tokens in prompt")
    completion_tokens: Optional[int] = Field(default=None, description="Number of tokens in completion")
    finish_reason: Optional[str] = Field(default=None, description="Reason for generation termination")


class BaseLLMProvider(ABC):
    """Abstract base provider contract for LLM text generation."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the LLM provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying LLM model."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generates completion text for a prompt with optional system instruction and parameters."""
        pass
