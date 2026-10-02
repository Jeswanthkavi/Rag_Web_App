"""Evaluation runner executing retrieval-only and end-to-end RAG evaluation sweeps."""

import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Set

from backend.app.evaluation.citation_metrics import compute_citation_metrics
from backend.app.evaluation.config import EvaluationConfig
from backend.app.evaluation.dataset import EvaluationDataset
from backend.app.evaluation.models import (
    EvaluationReport,
    EvaluationSample,
    GenerationMetrics,
    LatencyMetrics,
    RetrievalMetrics,
    SampleEvaluationResult,
)
from backend.app.evaluation.rag_metrics import BaseJudgeProvider, DeterministicJudge
from backend.app.evaluation.retrieval_metrics import evaluate_sample_retrieval
from backend.app.rag.service import RAGService
from backend.app.retrieval.models import RetrievalResult


class EvaluationRunner:
    """Orchestrates structured evaluation across datasets and computes aggregate benchmarks."""

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        judge_provider: Optional[BaseJudgeProvider] = None,
    ):
        self.config = config or EvaluationConfig()
        self.judge = judge_provider or DeterministicJudge()

    def _compute_latency_summary(self, latencies: List[float]) -> Dict[str, float]:
        """Calculates descriptive statistics for execution latencies."""
        if not latencies:
            return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_val = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

        return {
            "mean_ms": round(statistics.mean(latencies), 2),
            "median_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(p95_val, 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
        }

    def evaluate_retrieval(
        self,
        dataset: EvaluationDataset,
        search_fn: Callable[[str], List[RetrievalResult]],
        k_values: Optional[Sequence[int]] = None,
    ) -> EvaluationReport:
        """Runs retrieval-only evaluation benchmarking precision, recall, MRR, and latency."""
        effective_k = k_values or self.config.k_values
        sample_results: List[SampleEvaluationResult] = []
        total_latencies: List[float] = []

        for sample in dataset.samples:
            start_t = time.perf_counter()
            retrieved_chunks = search_fn(sample.query)
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            total_latencies.append(duration_ms)

            retrieved_ids = [
                c.chunk_id if hasattr(c, "chunk_id") else str(c)
                for c in retrieved_chunks
            ]
            expected_set = set(sample.expected_chunk_ids)

            ret_metrics = evaluate_sample_retrieval(
                retrieved=retrieved_ids,
                relevant=expected_set,
                k_values=effective_k,
            )

            sample_results.append(
                SampleEvaluationResult(
                    sample_id=sample.sample_id,
                    query=sample.query,
                    retrieved_chunk_ids=retrieved_ids,
                    retrieval_metrics=ret_metrics,
                    generated_answer=None,
                    generation_metrics=None,
                    latency=LatencyMetrics(
                        retrieval_latency_ms=round(duration_ms, 2),
                        total_latency_ms=round(duration_ms, 2),
                    ),
                )
            )

        # Compute Aggregates
        mean_p_at_k: Dict[int, float] = {}
        mean_r_at_k: Dict[int, float] = {}
        for k in effective_k:
            p_vals = [s.retrieval_metrics.precision_at_k.get(k, 0.0) for s in sample_results]
            r_vals = [s.retrieval_metrics.recall_at_k.get(k, 0.0) for s in sample_results]
            mean_p_at_k[k] = round(statistics.mean(p_vals), 4) if p_vals else 0.0
            mean_r_at_k[k] = round(statistics.mean(r_vals), 4) if r_vals else 0.0

        mrr_vals = [s.retrieval_metrics.mrr for s in sample_results]
        ctx_vals = [s.retrieval_metrics.context_relevance for s in sample_results]

        return EvaluationReport(
            dataset_name=dataset.name,
            total_samples=len(dataset),
            mean_precision_at_k=mean_p_at_k,
            mean_recall_at_k=mean_r_at_k,
            mrr=round(statistics.mean(mrr_vals), 4) if mrr_vals else 0.0,
            mean_context_relevance=round(statistics.mean(ctx_vals), 4) if ctx_vals else 0.0,
            latency_summary=self._compute_latency_summary(total_latencies),
            sample_results=sample_results,
        )

    def evaluate_rag(
        self,
        dataset: EvaluationDataset,
        rag_service: RAGService,
        user_id: str,
        k_values: Optional[Sequence[int]] = None,
    ) -> EvaluationReport:
        """Runs full end-to-end RAG evaluation benchmarking retrieval, answer relevance, faithfulness, and citations."""
        effective_k = k_values or self.config.k_values
        sample_results: List[SampleEvaluationResult] = []
        total_latencies: List[float] = []

        for sample in dataset.samples:
            start_t = time.perf_counter()
            response = rag_service.generate_answer(
                query=sample.query,
                user_id=user_id,
            )
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            total_latencies.append(duration_ms)

            # Citations chunk IDs
            cited_chunk_ids = [c.chunk_id for c in response.citations]
            expected_set = set(sample.expected_chunk_ids)

            # Evaluate retrieval component using cited/retrieved chunks
            ret_metrics = evaluate_sample_retrieval(
                retrieved=cited_chunk_ids,
                relevant=expected_set,
                k_values=effective_k,
            )

            # Evaluate generation metrics
            ans_rel, faith = self.judge.evaluate(
                query=sample.query,
                generated_answer=response.answer,
                expected_answer=sample.expected_answer,
                retrieved_contexts=[],
            )

            cit_metrics_dict = compute_citation_metrics(
                generated_citations=cited_chunk_ids,
                expected_citations=sample.expected_citations,
                total_retrieved_count=response.retrieved_context_count,
            )

            gen_metrics = GenerationMetrics(
                answer_relevance=round(ans_rel, 4),
                faithfulness=round(faith, 4),
                valid_citation_rate=round(cit_metrics_dict["valid_citation_rate"], 4),
                citation_precision=round(cit_metrics_dict["citation_precision"], 4),
                citation_coverage=round(cit_metrics_dict["citation_coverage"], 4),
            )

            sample_results.append(
                SampleEvaluationResult(
                    sample_id=sample.sample_id,
                    query=sample.query,
                    retrieved_chunk_ids=cited_chunk_ids,
                    retrieval_metrics=ret_metrics,
                    generated_answer=response.answer,
                    generation_metrics=gen_metrics,
                    latency=LatencyMetrics(
                        total_latency_ms=round(duration_ms, 2),
                    ),
                )
            )

        # Compute Aggregates
        mean_p_at_k: Dict[int, float] = {}
        mean_r_at_k: Dict[int, float] = {}
        for k in effective_k:
            p_vals = [s.retrieval_metrics.precision_at_k.get(k, 0.0) for s in sample_results]
            r_vals = [s.retrieval_metrics.recall_at_k.get(k, 0.0) for s in sample_results]
            mean_p_at_k[k] = round(statistics.mean(p_vals), 4) if p_vals else 0.0
            mean_r_at_k[k] = round(statistics.mean(r_vals), 4) if r_vals else 0.0

        mrr_vals = [s.retrieval_metrics.mrr for s in sample_results]
        ctx_vals = [s.retrieval_metrics.context_relevance for s in sample_results]

        ans_rel_vals = [s.generation_metrics.answer_relevance for s in sample_results if s.generation_metrics]
        faith_vals = [s.generation_metrics.faithfulness for s in sample_results if s.generation_metrics]
        cit_prec_vals = [s.generation_metrics.citation_precision for s in sample_results if s.generation_metrics]
        cit_cov_vals = [s.generation_metrics.citation_coverage for s in sample_results if s.generation_metrics]
        valid_cit_vals = [s.generation_metrics.valid_citation_rate for s in sample_results if s.generation_metrics]

        return EvaluationReport(
            dataset_name=dataset.name,
            total_samples=len(dataset),
            mean_precision_at_k=mean_p_at_k,
            mean_recall_at_k=mean_r_at_k,
            mrr=round(statistics.mean(mrr_vals), 4) if mrr_vals else 0.0,
            mean_context_relevance=round(statistics.mean(ctx_vals), 4) if ctx_vals else 0.0,
            mean_answer_relevance=round(statistics.mean(ans_rel_vals), 4) if ans_rel_vals else None,
            mean_faithfulness=round(statistics.mean(faith_vals), 4) if faith_vals else None,
            mean_citation_precision=round(statistics.mean(cit_prec_vals), 4) if cit_prec_vals else None,
            mean_citation_coverage=round(statistics.mean(cit_cov_vals), 4) if cit_cov_vals else None,
            mean_valid_citation_rate=round(statistics.mean(valid_cit_vals), 4) if valid_cit_vals else None,
            latency_summary=self._compute_latency_summary(total_latencies),
            sample_results=sample_results,
        )
