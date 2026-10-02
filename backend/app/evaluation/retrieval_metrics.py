"""Evaluation metrics for retrieval quality: Precision@K, Recall@K, MRR, Context Relevance."""

from typing import Dict, List, Sequence, Set
from backend.app.evaluation.models import RetrievalMetrics


def compute_precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Computes Precision@K: (relevant items in top K) / K.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.
        
    Returns:
        Float score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0

    top_k_items = retrieved[:k]
    relevant_retrieved = sum(1 for item in top_k_items if item in relevant)
    return float(relevant_retrieved) / float(k)


def compute_recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Computes Recall@K: (relevant items in top K) / (total relevant items).
    
    If no relevant items exist:
    - Returns 1.0 if retrieved is also empty, else 0.0.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.
        
    Returns:
        Float score between 0.0 and 1.0.
    """
    if not relevant:
        return 1.0 if not retrieved else 0.0
    if k <= 0:
        return 0.0

    top_k_items = retrieved[:k]
    relevant_retrieved = sum(1 for item in top_k_items if item in relevant)
    return float(relevant_retrieved) / float(len(relevant))


def compute_mrr(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """Computes Mean Reciprocal Rank (MRR) for a single query: 1 / rank_of_first_relevant.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant: Set of ground-truth relevant chunk IDs.
        
    Returns:
        Reciprocal rank float [0.0 - 1.0] (0.0 if no relevant items found).
    """
    if not relevant or not retrieved:
        return 0.0

    for rank_idx, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / float(rank_idx)

    return 0.0


def compute_context_relevance(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """Computes overall context relevance: (relevant retrieved items) / (total retrieved items)."""
    if not retrieved:
        return 0.0
    if not relevant:
        return 0.0

    relevant_count = sum(1 for item in retrieved if item in relevant)
    return float(relevant_count) / float(len(retrieved))


def evaluate_sample_retrieval(
    retrieved: Sequence[str],
    relevant: Set[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    """Computes all retrieval metrics for a single evaluation query."""
    p_at_k: Dict[int, float] = {}
    r_at_k: Dict[int, float] = {}

    for k in k_values:
        p_at_k[k] = compute_precision_at_k(retrieved, relevant, k)
        r_at_k[k] = compute_recall_at_k(retrieved, relevant, k)

    mrr_val = compute_mrr(retrieved, relevant)
    ctx_rel = compute_context_relevance(retrieved, relevant)

    return RetrievalMetrics(
        precision_at_k=p_at_k,
        recall_at_k=r_at_k,
        mrr=mrr_val,
        context_relevance=ctx_rel,
    )
