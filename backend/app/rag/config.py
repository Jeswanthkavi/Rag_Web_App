"""RAG generation configuration and settings loader."""

from dataclasses import dataclass
from typing import Optional
from backend.app.core.config import Settings, get_settings


DEFAULT_NO_CONTEXT_MESSAGE = (
    "I do not have enough information in the provided documents to answer this question."
)


@dataclass(frozen=True)
class RAGConfig:
    """Runtime configuration for grounded RAG generation."""
    top_k: int = 5
    similarity_threshold: Optional[float] = 0.5
    temperature: float = 0.2
    max_tokens: int = 2048
    max_query_length: int = 1000
    no_context_message: str = DEFAULT_NO_CONTEXT_MESSAGE

    def __post_init__(self):
        if self.top_k < 1 or self.top_k > 100:
            raise ValueError(f"top_k must be between 1 and 100, got {self.top_k}")
        if self.similarity_threshold is not None:
            if not (0.0 <= self.similarity_threshold <= 1.0):
                raise ValueError(
                    f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
                )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if not (64 <= self.max_tokens <= 8192):
            raise ValueError(f"max_tokens must be between 64 and 8192, got {self.max_tokens}")
        if not (10 <= self.max_query_length <= 10000):
            raise ValueError(
                f"max_query_length must be between 10 and 10000, got {self.max_query_length}"
            )

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "RAGConfig":
        """Builds RAGConfig from application Pydantic settings with optional overrides."""
        s = settings or get_settings()
        k = top_k if top_k is not None else s.retrieval.retrieval_top_k
        thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else s.retrieval.score_threshold
        )
        temp = temperature if temperature is not None else s.gemini.gemini_temperature
        tokens = max_tokens if max_tokens is not None else s.gemini.gemini_max_tokens

        return cls(
            top_k=k,
            similarity_threshold=thresh,
            temperature=temp,
            max_tokens=tokens,
        )
