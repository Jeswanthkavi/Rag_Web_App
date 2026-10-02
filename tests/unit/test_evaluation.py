"""Unit tests for Phase 18 Evaluation Framework.

Tests:
1. Retrieval metrics (Precision@K, Recall@K, MRR, Context Relevance)
2. Zero-relevant items edge case handling
3. Citation metrics (valid citation rate, precision, coverage)
4. Groundedness & generation metrics (Answer Relevance, Faithfulness)
5. Dataset loading (synthetic, JSON, JSONL)
6. EvaluationRunner for retrieval-only and end-to-end evaluation
7. Formatted summary reports
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.evaluation.citation_metrics import compute_citation_metrics
from app.evaluation.config import EvaluationConfig
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.models import EvaluationSample
from app.evaluation.rag_metrics import (
    DeterministicJudge,
    FakeJudge,
    compute_answer_relevance,
    compute_faithfulness,
)
from app.evaluation.report import format_evaluation_summary
from app.evaluation.retrieval_metrics import (
    compute_context_relevance,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_sample_retrieval,
)
from app.evaluation.runner import EvaluationRunner
from app.retrieval.models import RetrievalResult


# ============================================================================
# Retrieval Metrics Tests
# ============================================================================


class TestRetrievalMetrics:
    def test_precision_at_k_basic(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = ["c1", "c3", "c99"]

        assert compute_precision_at_k(retrieved, relevant, k=1) == 1.0  # c1 in top 1 -> 1/1
        assert compute_precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)  # c1, c3 in top 3 -> 2/3
        assert compute_precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5)  # c1, c3 in top 5 -> 2/5

    def test_precision_at_k_zero_or_empty(self):
        assert compute_precision_at_k([], ["c1"], k=5) == 0.0
        assert compute_precision_at_k(["c1"], [], k=5) == 0.0
        assert compute_precision_at_k(["c1"], ["c1"], k=0) == 0.0

    def test_recall_at_k_basic(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = ["c1", "c3", "c7"]

        assert compute_recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)  # c1 found out of 3
        assert compute_recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)  # c1, c3 found out of 3
        assert compute_recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)  # c1, c3 found out of 3

    def test_recall_at_k_zero_relevant_items(self):
        # Edge case: no relevant items exist in ground truth
        assert compute_recall_at_k(["c1", "c2"], [], k=5) == 0.0
        assert compute_recall_at_k([], [], k=5) == 1.0

    def test_recall_at_k_perfect(self):
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c1", "c2"]
        assert compute_recall_at_k(retrieved, relevant, k=2) == 1.0

    def test_mrr_basic(self):
        # First relevant at rank 1
        assert compute_mrr(["c1", "c2", "c3"], ["c1"]) == 1.0
        # First relevant at rank 2
        assert compute_mrr(["c2", "c1", "c3"], ["c1"]) == 0.5
        # First relevant at rank 3
        assert compute_mrr(["c2", "c3", "c1"], ["c1"]) == pytest.approx(1 / 3)
        # No relevant found
        assert compute_mrr(["c2", "c3", "c4"], ["c1"]) == 0.0
        # Empty lists
        assert compute_mrr([], ["c1"]) == 0.0
        assert compute_mrr(["c1"], []) == 0.0

    def test_context_relevance(self):
        retrieved = ["c1", "c2", "c3", "c4"]
        relevant = ["c1", "c3"]
        # 2 out of 4 retrieved chunks are relevant
        assert compute_context_relevance(retrieved, relevant) == 0.5

        # Empty retrieved
        assert compute_context_relevance([], relevant) == 0.0

    def test_evaluate_sample_retrieval(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = ["c1", "c3", "c7"]

        metrics = evaluate_sample_retrieval(retrieved, relevant, k_values=[1, 3, 5])
        assert metrics.precision_at_k[1] == 1.0
        assert metrics.precision_at_k[3] == pytest.approx(2 / 3)
        assert metrics.precision_at_k[5] == pytest.approx(2 / 5)
        assert metrics.recall_at_k[1] == pytest.approx(1 / 3)
        assert metrics.recall_at_k[3] == pytest.approx(2 / 3)
        assert metrics.recall_at_k[5] == pytest.approx(2 / 3)
        assert metrics.mrr == 1.0
        assert metrics.context_relevance == pytest.approx(2 / 5)


# ============================================================================
# Citation Metrics Tests
# ============================================================================


class TestCitationMetrics:
    def test_citation_metrics_all_valid(self):
        citations = ["doc1.txt", "doc2.txt"]
        expected_sources = ["doc1.txt", "doc2.txt"]

        metrics = compute_citation_metrics(
            generated_citations=citations,
            expected_citations=expected_sources,
            total_retrieved_count=2,
        )
        assert metrics["valid_citation_rate"] == 1.0
        assert metrics["citation_precision"] == 1.0
        assert metrics["citation_coverage"] == 1.0

    def test_citation_metrics_partial_valid(self):
        citations = ["doc1.txt", "doc_unverified.txt"]
        expected_sources = ["doc1.txt", "doc2.txt"]

        metrics = compute_citation_metrics(
            generated_citations=citations,
            expected_citations=expected_sources,
            total_retrieved_count=2,
        )
        assert metrics["valid_citation_rate"] == 1.0
        assert metrics["citation_precision"] == 0.5  # 1/2 in expected
        assert metrics["citation_coverage"] == 0.5  # 1/2 expected cited

    def test_citation_metrics_no_citations(self):
        metrics = compute_citation_metrics(
            generated_citations=[],
            expected_citations=["doc1.txt"],
            total_retrieved_count=2,
        )
        assert metrics["valid_citation_rate"] == 0.0
        assert metrics["citation_precision"] == 0.0
        assert metrics["citation_coverage"] == 0.0

    def test_citation_metrics_no_expected_sources(self):
        citations = ["doc1.txt"]
        metrics = compute_citation_metrics(
            generated_citations=citations,
            expected_citations=[],
            total_retrieved_count=1,
        )
        assert metrics["valid_citation_rate"] == 1.0
        assert metrics["citation_precision"] == 1.0
        assert metrics["citation_coverage"] == 1.0


# ============================================================================
# Generation & Groundedness Metrics Tests
# ============================================================================


class TestGenerationMetrics:
    def test_deterministic_judge_answer_relevance(self):
        judge = DeterministicJudge()
        score = judge.evaluate_answer_relevance(
            query="What is the refund policy?",
            answer="The refund policy allows 30-day returns on all purchases.",
            expected_answer="Customers can request a refund within 30 days of purchase.",
        )
        assert score > 0.1


    def test_deterministic_judge_faithfulness(self):
        judge = DeterministicJudge()
        context = [
            "We offer a 30-day money-back guarantee for all software licenses.",
            "Items must be in original condition.",
        ]
        answer = "A 30-day money-back guarantee is available for software licenses."
        score = judge.evaluate_faithfulness(answer=answer, context_chunks=context)
        assert score > 0.5

    def test_faithfulness_hallucinated_answer(self):
        judge = DeterministicJudge()
        context = ["Apples and oranges are grown in temperate climates."]
        answer = "Quantum computers operate using superconducting qubits at cryogenic temperatures."
        score = judge.evaluate_faithfulness(answer=answer, context_chunks=context)
        assert score < 0.2

    def test_fake_judge(self):
        fake = FakeJudge(fixed_relevance=0.95, fixed_faithfulness=0.88)
        assert fake.evaluate_answer_relevance("q", "a") == 0.95
        assert fake.evaluate_faithfulness("a", ["c"]) == 0.88

    def test_empty_generation_inputs(self):
        assert compute_answer_relevance("", "") == 0.0
        assert compute_faithfulness("", []) == 0.0


# ============================================================================
# Dataset Loading & Management Tests
# ============================================================================


class TestEvaluationDataset:
    def test_create_synthetic(self):
        dataset = EvaluationDataset.create_synthetic()
        assert len(dataset) >= 5
        sample = dataset[0]
        assert sample is not None
        assert sample.sample_id
        assert sample.query
        assert len(sample.expected_chunk_ids) > 0

    def test_json_load_and_save(self):
        dataset = EvaluationDataset.create_synthetic()
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "eval_dataset.json"
            dataset.save_json(str(file_path))

            loaded = EvaluationDataset.from_json(str(file_path))
            assert len(loaded) == len(dataset)
            assert loaded[0].sample_id == dataset[0].sample_id
            assert loaded[0].query == dataset[0].query

    def test_jsonl_load(self):
        samples_data = [
            {
                "sample_id": "sample_1",
                "query": "What is Python?",
                "expected_chunk_ids": ["c1", "c2"],
                "expected_answer": "Python is a programming language.",
                "expected_citations": ["c1", "c2"],
            },
            {
                "sample_id": "sample_2",
                "query": "What is RAG?",
                "expected_chunk_ids": ["c3"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "eval_dataset.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for item in samples_data:
                    f.write(json.dumps(item) + "\n")

            loaded = EvaluationDataset.from_jsonl(str(file_path))
            assert len(loaded) == 2
            assert loaded[0].sample_id == "sample_1"
            assert loaded[1].expected_chunk_ids == ["c3"]


# ============================================================================
# Evaluation Runner Tests
# ============================================================================


class TestEvaluationRunner:
    def test_retrieval_only_runner(self):
        dataset = EvaluationDataset.create_synthetic()
        config = EvaluationConfig(k_values=[1, 3, 5])

        def mock_search_fn(query: str):
            if "battery" in query.lower():
                return ["chunk_battery_warranty_01", "chunk_noise_02", "chunk_noise_03"]
            elif "kubernetes" in query.lower():
                return ["chunk_k8s_autoscaling_02", "chunk_noise_01"]
            else:
                return ["chunk_solar_inverter_03", "chunk_noise_04"]

        runner = EvaluationRunner(config=config, judge_provider=DeterministicJudge())
        report = runner.evaluate_retrieval(
            dataset=dataset,
            search_fn=mock_search_fn,
        )

        assert report.total_samples == len(dataset)
        assert 1 in report.mean_precision_at_k
        assert 3 in report.mean_recall_at_k
        assert report.mrr > 0.0
        assert report.latency_summary["mean_ms"] >= 0.0

    def test_rag_end_to_end_runner(self):
        dataset = EvaluationDataset([
            EvaluationSample(
                sample_id="test_1",
                query="How long is data kept?",
                expected_chunk_ids=["c_ret_1"],
                expected_answer="Data is retained for 30 days after account deletion.",
                expected_citations=["c_ret_1"],
            )
        ])

        mock_rag_service = MagicMock()
        mock_response = MagicMock()
        mock_citation = MagicMock()
        mock_citation.chunk_id = "c_ret_1"
        mock_citation.verified = True
        mock_response.citations = [mock_citation]
        mock_response.answer = "Data is retained for 30 days after account deletion."
        mock_response.retrieved_context_count = 1
        mock_rag_service.generate_answer.return_value = mock_response

        runner = EvaluationRunner(
            config=EvaluationConfig(k_values=[1, 3, 5]),
            judge_provider=DeterministicJudge(),
        )
        report = runner.evaluate_rag(
            dataset=dataset,
            rag_service=mock_rag_service,
            user_id="user_test",
        )

        assert report.total_samples == 1
        assert report.mean_precision_at_k[1] == 1.0
        assert report.mean_recall_at_k[1] == 1.0
        assert report.mrr == 1.0
        assert report.mean_answer_relevance is not None and report.mean_answer_relevance > 0.8
        assert report.mean_faithfulness is not None
        assert report.mean_valid_citation_rate == 1.0

    def test_format_evaluation_summary(self):
        dataset = EvaluationDataset.create_synthetic()
        runner = EvaluationRunner(judge_provider=FakeJudge(fixed_relevance=0.9, fixed_faithfulness=0.9))

        def dummy_search_fn(query: str):
            return ["chunk_battery_warranty_01"]

        report = runner.evaluate_retrieval(
            dataset=dataset,
            search_fn=dummy_search_fn,
        )

        summary_text = format_evaluation_summary(report)
        assert "ADVANCED RAG EVALUATION BENCHMARK REPORT" in summary_text
        assert "Precision" in summary_text
        assert "MRR" in summary_text
        assert "Latency Summary" in summary_text


