"""Reciprocal Rank Fusion (RRF) for combining dense semantic and lexical retrieval candidate lists."""

import logging
from typing import Dict, List, Optional, Tuple
from backend.app.hybrid_retrieval.models import HybridCandidate
from backend.app.retrieval.models import RetrievalResult

logger = logging.getLogger("app.hybrid_retrieval.fusion")


def reciprocal_rank_fusion(
    dense_results: List[RetrievalResult],
    lexical_results: List[RetrievalResult],
    rrf_k: int = 60,
    limit: Optional[int] = None,
) -> Tuple[List[RetrievalResult], List[HybridCandidate]]:
    """Fuses dense and lexical candidate lists using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_score(chunk) = sum( 1 / (rrf_k + rank_i) ) for each system where chunk appears
        
    Guarantees:
    1. Chunk Deduplication: Chunks appearing in both dense and lexical results are merged into
       a single candidate with combined RRF score.
    2. Rank Preservation: Preserves dense_rank, lexical_rank, dense_score, and lexical_score.
    3. Deterministic Sorting: Fused candidates are sorted descending by fusion_score.
    4. Slicing: Returns at most `limit` items.
    
    Args:
        dense_results: Ranked list of results from dense vector retrieval.
        lexical_results: Ranked list of results from lexical/BM25 retrieval.
        rrf_k: Smoothing constant (default 60).
        limit: Optional maximum candidate count.
        
    Returns:
        Tuple of (List[RetrievalResult], List[HybridCandidate]).
    """
    candidates_by_id: Dict[str, HybridCandidate] = {}

    # 1. Process dense candidates
    for rank_idx, r in enumerate(dense_results, start=1):
        rrf_score = 1.0 / (rrf_k + rank_idx)
        candidates_by_id[r.chunk_id] = HybridCandidate(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            user_id=r.user_id,
            chunk_index=r.chunk_index,
            chunk_text=r.chunk_text,
            page_number=r.page_number,
            filename=r.filename,
            file_type=r.file_type,
            token_count=r.token_count,
            dense_score=r.score,
            dense_rank=rank_idx,
            lexical_score=None,
            lexical_rank=None,
            fusion_score=rrf_score,
            rank=1,
        )

    # 2. Process lexical candidates and fuse scores
    for rank_idx, r in enumerate(lexical_results, start=1):
        rrf_score = 1.0 / (rrf_k + rank_idx)
        if r.chunk_id in candidates_by_id:
            # Chunk present in both retrieval branches: update and sum RRF scores
            cand = candidates_by_id[r.chunk_id]
            cand.lexical_score = r.score
            cand.lexical_rank = rank_idx
            cand.fusion_score += rrf_score
        else:
            # Chunk present only in lexical branch
            candidates_by_id[r.chunk_id] = HybridCandidate(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                user_id=r.user_id,
                chunk_index=r.chunk_index,
                chunk_text=r.chunk_text,
                page_number=r.page_number,
                filename=r.filename,
                file_type=r.file_type,
                token_count=r.token_count,
                dense_score=None,
                dense_rank=None,
                lexical_score=r.score,
                lexical_rank=rank_idx,
                fusion_score=rrf_score,
                rank=1,
            )

    # 3. Sort descending by fusion score (break ties by dense_score if available)
    fused_list = list(candidates_by_id.values())
    fused_list.sort(
        key=lambda c: (c.fusion_score, c.dense_score or 0.0, c.lexical_score or 0.0),
        reverse=True,
    )

    # 4. Limit to requested candidate count
    sliced_candidates = fused_list[:limit] if limit is not None else fused_list

    # 5. Assign 1-indexed final ranks and convert to RetrievalResult
    final_candidates: List[HybridCandidate] = []
    final_retrieval_results: List[RetrievalResult] = []

    for rank_idx, cand in enumerate(sliced_candidates, start=1):
        cand.rank = rank_idx
        final_candidates.append(cand)
        final_retrieval_results.append(cand.to_retrieval_result())

    return final_retrieval_results, final_candidates
