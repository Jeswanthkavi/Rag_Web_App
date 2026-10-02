"""Configuration dataclass and loader for candidate chunk reranking."""

from dataclasses import dataclass
from typing import Optional
from backend.app.core.config import Settings, get_settings


@dataclass(frozen=True)
class RerankingConfig:
    """Runtime configuration for candidate chunk reranking."""
    enabled: bool = True
    provider: str = "local"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_top_k: int = 20
    final_top_k: int = 5
    batch_size: int = 32

    def __post_init__(self):
        if self.retrieval_top_k < 1 or self.retrieval_top_k > 200:
            raise ValueError(f"retrieval_top_k must be between 1 and 200, got {self.retrieval_top_k}")
        if self.final_top_k < 1 or self.final_top_k > 100:
            raise ValueError(f"final_top_k must be between 1 and 100, got {self.final_top_k}")
        if self.final_top_k > self.retrieval_top_k:
            raise ValueError(
                f"final_top_k ({self.final_top_k}) cannot exceed retrieval_top_k ({self.retrieval_top_k})"
            )
        if self.batch_size < 1 or self.batch_size > 256:
            raise ValueError(f"batch_size must be between 1 and 256, got {self.batch_size}")

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        enabled: Optional[bool] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        retrieval_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> "RerankingConfig":
        """Builds RerankingConfig from application Pydantic settings with optional overrides."""
        s = settings or get_settings()
        r_cfg = s.retrieval
        return cls(
            enabled=enabled if enabled is not None else r_cfg.enable_reranking,
            provider=provider if provider is not None else "local",
            model_name=model_name if model_name is not None else r_cfg.reranking_model,
            retrieval_top_k=retrieval_top_k if retrieval_top_k is not None else r_cfg.retrieval_top_k,
            final_top_k=final_top_k if final_top_k is not None else r_cfg.reranking_top_k,
            batch_size=batch_size if batch_size is not None else 32,
        )
