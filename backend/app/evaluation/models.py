"""Data models and metrics schemas for retrieval and grounded RAG evaluation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EvaluationSample(BaseModel):
    """An individual evaluation test case."""
    model_config = ConfigDict(from_attributes=True)

    sample_id: str = Field(..., description="Unique test case identifier")
    query: str = Field(..., description="Evaluation search query or question")
    expected_chunk_ids: List[str] = Field(default_factory=list, description="Ground truth relevant chunk IDs")
    expected_answer: Optional[str] = Field(default=None, description="Ground truth answer or expected factual statements")
    expected_citations: List[str] = Field(default_factory=list, description="Expected citation chunk IDs or source identifiers")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata for segmentation/tagging")


class RetrievalMetrics(BaseModel):
    """Retrieval performance metrics."""
    model_config = ConfigDict(from_attributes=True)

    precision_at_k: Dict[int, float] = Field(default_factory=dict, description="Precision@K for various K values")
    recall_at_k: Dict[int, float] = Field(default_factory=dict, description="Recall@K for various K values")
    mrr: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    context_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Proportion of retrieved chunks matching relevant set")


class GenerationMetrics(BaseModel):
    """Answer generation and groundedness metrics."""
    model_config = ConfigDict(from_attributes=True)

    answer_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Semantic/lexical relevance to expected answer")
    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0, description="Groundedness of answer in retrieved context")
    valid_citation_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of citations that map to real retrieved sources")
    citation_precision: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of generated citations that match expected chunks")
    citation_coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of expected relevant chunks cited in the answer")


class LatencyMetrics(BaseModel):
    """Execution timing measurements in milliseconds."""
    model_config = ConfigDict(from_attributes=True)

    retrieval_latency_ms: float = Field(default=0.0, ge=0.0)
    reranking_latency_ms: Optional[float] = Field(default=None, ge=0.0)
    generation_latency_ms: Optional[float] = Field(default=None, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)


class SampleEvaluationResult(BaseModel):
    """Complete evaluation result for a single sample."""
    model_config = ConfigDict(from_attributes=True)

    sample_id: str
    query: str
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    retrieval_metrics: RetrievalMetrics
    generated_answer: Optional[str] = None
    generation_metrics: Optional[GenerationMetrics] = None
    latency: LatencyMetrics


class EvaluationReport(BaseModel):
    """Aggregated evaluation report across an evaluation dataset."""
    model_config = ConfigDict(from_attributes=True)

    dataset_name: str
    total_samples: int = Field(ge=0)
    mean_precision_at_k: Dict[int, float] = Field(default_factory=dict)
    mean_recall_at_k: Dict[int, float] = Field(default_factory=dict)
    mrr: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_context_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_answer_relevance: Optional[float] = None
    mean_faithfulness: Optional[float] = None
    mean_valid_citation_rate: Optional[float] = None
    mean_citation_precision: Optional[float] = None
    mean_citation_coverage: Optional[float] = None
    latency_summary: Dict[str, Any] = Field(default_factory=dict)
    sample_results: List[SampleEvaluationResult] = Field(default_factory=list)
