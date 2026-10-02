"""Configuration loader and dataclass for hybrid dense and lexical retrieval."""

from dataclasses import dataclass
from typing import Optional
from backend.app.core.config import Settings, get_settings


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Runtime configuration parameters for hybrid retrieval and fusion."""
    enabled: bool = True
    dense_top_k: int = 20
    lexical_top_k: int = 20
    fusion_candidate_limit: int = 20
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def __post_init__(self):
        if self.dense_top_k < 1 or self.dense_top_k > 200:
            raise ValueError(f"dense_top_k must be between 1 and 200, got {self.dense_top_k}")
        if self.lexical_top_k < 1 or self.lexical_top_k > 200:
            raise ValueError(f"lexical_top_k must be between 1 and 200, got {self.lexical_top_k}")
        if self.fusion_candidate_limit < 1 or self.fusion_candidate_limit > 200:
            raise ValueError(f"fusion_candidate_limit must be between 1 and 200, got {self.fusion_candidate_limit}")
        if self.rrf_k < 1 or self.rrf_k > 500:
            raise ValueError(f"rrf_k must be between 1 and 500, got {self.rrf_k}")
        if not (0.1 <= self.bm25_k1 <= 10.0):
            raise ValueError(f"bm25_k1 must be between 0.1 and 10.0, got {self.bm25_k1}")
        if not (0.0 <= self.bm25_b <= 1.0):
            raise ValueError(f"bm25_b must be between 0.0 and 1.0, got {self.bm25_b}")

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        enabled: Optional[bool] = None,
        dense_top_k: Optional[int] = None,
        lexical_top_k: Optional[int] = None,
        fusion_candidate_limit: Optional[int] = None,
        rrf_k: Optional[int] = None,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
    ) -> "HybridRetrievalConfig":
        """Builds HybridRetrievalConfig from application Pydantic settings with optional overrides."""
        s = settings or get_settings()
        h_cfg = s.hybrid_retrieval
        return cls(
            enabled=enabled if enabled is not None else h_cfg.enable_hybrid_retrieval,
            dense_top_k=dense_top_k if dense_top_k is not None else h_cfg.dense_top_k,
            lexical_top_k=lexical_top_k if lexical_top_k is not None else h_cfg.lexical_top_k,
            fusion_candidate_limit=fusion_candidate_limit if fusion_candidate_limit is not None else h_cfg.fusion_candidate_limit,
            rrf_k=rrf_k if rrf_k is not None else h_cfg.rrf_k,
            bm25_k1=bm25_k1 if bm25_k1 is not None else h_cfg.bm25_k1,
            bm25_b=bm25_b if bm25_b is not None else h_cfg.bm25_b,
        )
