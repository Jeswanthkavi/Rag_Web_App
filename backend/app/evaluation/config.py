"""Configuration parameters for evaluation runs."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class EvaluationConfig:
    """Runtime configuration for running evaluation sweeps."""
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    evaluate_generation: bool = True
    judge_model: str = "deterministic"
    dataset_name: str = "synthetic_evaluation_dataset"

    def __post_init__(self):
        if not self.k_values:
            raise ValueError("k_values cannot be empty.")
        for k in self.k_values:
            if k < 1:
                raise ValueError(f"k must be >= 1, got {k}")
