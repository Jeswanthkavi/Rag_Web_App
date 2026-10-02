"""Evaluation runner script for Advanced RAG benchmark reporting."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.evaluation.dataset import EvaluationDataset
from backend.app.evaluation.report import format_evaluation_summary
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.retrieval.models import RetrievalResult


def synthetic_retrieval_fn(query: str) -> list[RetrievalResult]:
    """Deterministic retrieval function for evaluation testing."""
    q_lower = query.lower()
    results = []
    if "battery" in q_lower or "warranty" in q_lower:
        results.append(
            RetrievalResult(
                chunk_id="chunk_battery_warranty_01",
                document_id="doc_battery",
                user_id="eval_user",
                chunk_index=0,
                chunk_text="The lithium battery includes a 10-year replacement warranty.",
                filename="battery_guide.pdf",
                file_type=".pdf",
                score=0.92,
                rank=1,
            )
        )
    if "kubernetes" in q_lower or "autoscaler" in q_lower or "auto-scaler" in q_lower:
        results.append(
            RetrievalResult(
                chunk_id="chunk_k8s_autoscaling_02",
                document_id="doc_k8s",
                user_id="eval_user",
                chunk_index=0,
                chunk_text="Kubernetes Horizontal Pod Autoscaler handles traffic spikes.",
                filename="k8s.pdf",
                file_type=".pdf",
                score=0.88,
                rank=1,
            )
        )
    if "return" in q_lower or "refund" in q_lower:
        results.extend([
            RetrievalResult(
                chunk_id="chunk_refund_policy_01",
                document_id="doc_refund",
                user_id="eval_user",
                chunk_index=0,
                chunk_text="Return items within 30 days in original packaging.",
                filename="refund.pdf",
                file_type=".pdf",
                score=0.95,
                rank=1,
            ),
            RetrievalResult(
                chunk_id="chunk_refund_timeline_02",
                document_id="doc_refund",
                user_id="eval_user",
                chunk_index=1,
                chunk_text="Refunds are processed within 5 business days.",
                filename="refund.pdf",
                file_type=".pdf",
                score=0.89,
                rank=2,
            ),
        ])
    if "postgresql" in q_lower or "failover" in q_lower:
        results.append(
            RetrievalResult(
                chunk_id="chunk_db_failover_01",
                document_id="doc_db",
                user_id="eval_user",
                chunk_index=0,
                chunk_text="PostgreSQL uses PgBouncer with asynchronous replica promotion and RPO < 5 min.",
                filename="db.pdf",
                file_type=".pdf",
                score=0.91,
                rank=1,
            )
        )
    if "solar" in q_lower or "inverter" in q_lower:
        results.append(
            RetrievalResult(
                chunk_id="chunk_solar_inverter_03",
                document_id="doc_solar",
                user_id="eval_user",
                chunk_index=0,
                chunk_text="Solar inverter operates between -20C and +50C.",
                filename="solar.pdf",
                file_type=".pdf",
                score=0.87,
                rank=1,
            )
        )
    return results


def main():
    print("Loading synthetic evaluation dataset...")
    dataset = EvaluationDataset.from_synthetic()
    print(f"Loaded {len(dataset)} evaluation samples.")

    runner = EvaluationRunner()
    print("Running retrieval evaluation benchmarks...")
    report = runner.evaluate_retrieval(
        dataset=dataset,
        search_fn=synthetic_retrieval_fn,
        k_values=[1, 3, 5, 10],
    )

    formatted = format_evaluation_summary(report)
    print("\n" + formatted + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
