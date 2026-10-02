"""Evaluation framework module for benchmarking retrieval, answer relevance, faithfulness, citations, and latency."""

from backend.app.evaluation.citation_metrics import compute_citation_metrics
from backend.app.evaluation.config import EvaluationConfig
from backend.app.evaluation.dataset import SYNTHETIC_EVALUATION_DATA, EvaluationDataset
from backend.app.evaluation.models import (
    EvaluationReport,
    EvaluationSample,
    GenerationMetrics,
    LatencyMetrics,
    RetrievalMetrics,
    SampleEvaluationResult,
)
from backend.app.evaluation.rag_metrics import (
    BaseJudgeProvider,
    DeterministicJudge,
    FakeJudge,
    compute_answer_relevance,
    compute_faithfulness,
)
from backend.app.evaluation.report import format_evaluation_summary
from backend.app.evaluation.retrieval_metrics import (
    compute_context_relevance,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_sample_retrieval,
)
from backend.app.evaluation.runner import EvaluationRunner

__all__ = [
    "BaseJudgeProvider",
    "DeterministicJudge",
    "EvaluationConfig",
    "EvaluationDataset",
    "EvaluationReport",
    "EvaluationRunner",
    "EvaluationSample",
    "FakeJudge",
    "GenerationMetrics",
    "LatencyMetrics",
    "RetrievalMetrics",
    "SYNTHETIC_EVALUATION_DATA",
    "SampleEvaluationResult",
    "compute_answer_relevance",
    "compute_citation_metrics",
    "compute_context_relevance",
    "compute_faithfulness",
    "compute_mrr",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "evaluate_sample_retrieval",
    "format_evaluation_summary",
]
