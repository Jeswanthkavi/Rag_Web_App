"""Evaluation report formatter for console and markdown presentation."""

from backend.app.evaluation.models import EvaluationReport


def format_evaluation_summary(report: EvaluationReport) -> str:
    """Formats an EvaluationReport into a readable text summary."""
    lines = [
        "=" * 65,
        f" ADVANCED RAG EVALUATION BENCHMARK REPORT: {report.dataset_name}",
        "=" * 65,
        f" Total Samples Evaluated: {report.total_samples}",
        "",
        "--- Retrieval Benchmark Metrics ---",
    ]

    for k, p in sorted(report.mean_precision_at_k.items()):
        r = report.mean_recall_at_k.get(k, 0.0)
        lines.append(f"  Precision@{k:02d}: {p:.4f}  |  Recall@{k:02d}: {r:.4f}")

    lines.append(f"  MRR (Mean Reciprocal Rank): {report.mrr:.4f}")
    lines.append(f"  Context Relevance:          {report.mean_context_relevance:.4f}")
    lines.append("")

    if report.mean_answer_relevance is not None:
        lines.extend([
            "--- Generation & Groundedness Metrics ---",
            f"  Answer Relevance:           {report.mean_answer_relevance:.4f}",
            f"  Faithfulness / Groundedness:{report.mean_faithfulness:.4f}",
            f"  Valid Citation Rate:        {report.mean_valid_citation_rate:.4f}",
            f"  Citation Precision:         {report.mean_citation_precision:.4f}",
            f"  Citation Coverage:          {report.mean_citation_coverage:.4f}",
            "",
        ])

    lines.extend([
        "--- Execution Latency Summary (ms) ---",
        f"  Mean:   {report.latency_summary.get('mean_ms', 0.0):.2f} ms",
        f"  Median: {report.latency_summary.get('median_ms', 0.0):.2f} ms",
        f"  p95:    {report.latency_summary.get('p95_ms', 0.0):.2f} ms",
        f"  Min:    {report.latency_summary.get('min_ms', 0.0):.2f} ms",
        f"  Max:    {report.latency_summary.get('max_ms', 0.0):.2f} ms",
        "=" * 65,
    ])

    return "\n".join(lines)
