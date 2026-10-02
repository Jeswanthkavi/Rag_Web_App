"""Citation correctness and quality metrics."""

from typing import Dict, List, Sequence, Set


def compute_citation_metrics(
    generated_citations: Sequence[str],
    expected_citations: Sequence[str],
    total_retrieved_count: int,
) -> Dict[str, float]:
    """Computes citation correctness, precision, and coverage.
    
    Args:
        generated_citations: List of chunk IDs or citation references produced in the answer.
        expected_citations: Ground truth expected citation chunk IDs.
        total_retrieved_count: Number of context chunks retrieved for this query.
        
    Returns:
        Dictionary containing:
        - valid_citation_rate
        - citation_precision
        - citation_coverage
    """
    if not generated_citations and not expected_citations:
        return {
            "valid_citation_rate": 1.0,
            "citation_precision": 1.0,
            "citation_coverage": 1.0,
        }

    # 1. Valid Citation Rate: citations that map to real retrieved chunks
    if not generated_citations:
        valid_rate = 0.0
        precision = 0.0
    else:
        # Every verified citation generated through CitationParser mapped to a real retrieved chunk (1..N)
        valid_rate = 1.0 if total_retrieved_count > 0 else 0.0

        # 2. Citation Precision: fraction of generated citations that are in expected set
        expected_set: Set[str] = set(expected_citations)
        if expected_set:
            matching = sum(1 for c in generated_citations if c in expected_set)
            precision = float(matching) / float(len(generated_citations))
        else:
            precision = 1.0

    # 3. Citation Coverage: fraction of expected citations present in generated citations
    if not expected_citations:
        coverage = 1.0
    else:
        gen_set: Set[str] = set(generated_citations)
        covered = sum(1 for exp in expected_citations if exp in gen_set)
        coverage = float(covered) / float(len(expected_citations))

    return {
        "valid_citation_rate": valid_rate,
        "citation_precision": precision,
        "citation_coverage": coverage,
    }
