"""Configuration loader and dataclass for conversational query rewriting."""

from dataclasses import dataclass
from typing import Optional
from backend.app.core.config import Settings, get_settings


@dataclass(frozen=True)
class QueryRewritingConfig:
    """Runtime configuration parameters for conversational query rewriting."""
    enabled: bool = True
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    temperature: float = 0.0
    max_history_messages: int = 5
    max_query_length: int = 500
    timeout: float = 10.0

    def __post_init__(self):
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {self.temperature}")
        if not (1 <= self.max_history_messages <= 50):
            raise ValueError(f"max_history_messages must be between 1 and 50, got {self.max_history_messages}")
        if not (10 <= self.max_query_length <= 5000):
            raise ValueError(f"max_query_length must be between 10 and 5000, got {self.max_query_length}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        enabled: Optional[bool] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_history_messages: Optional[int] = None,
        max_query_length: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> "QueryRewritingConfig":
        """Builds QueryRewritingConfig from application Pydantic settings with optional overrides."""
        s = settings or get_settings()
        q_cfg = s.query_rewriting
        return cls(
            enabled=enabled if enabled is not None else q_cfg.enable_query_rewriting,
            model_name=model_name if model_name is not None else q_cfg.query_rewriter_model,
            temperature=temperature if temperature is not None else q_cfg.query_rewriter_temperature,
            max_history_messages=max_history_messages if max_history_messages is not None else q_cfg.query_rewriter_max_history_messages,
            max_query_length=max_query_length if max_query_length is not None else q_cfg.query_rewriter_max_query_length,
            timeout=timeout if timeout is not None else q_cfg.query_rewriter_timeout,
        )
