"""Hybrid retrieval module combining dense semantic vectors and lexical BM25 with Reciprocal Rank Fusion."""

from backend.app.hybrid_retrieval.base import (
    BaseLexicalRetriever,
    HybridRetrievalError,
    LexicalRetrievalError,
)
from backend.app.hybrid_retrieval.bm25 import BM25Okapi, tokenize
from backend.app.hybrid_retrieval.config import HybridRetrievalConfig
from backend.app.hybrid_retrieval.fake_provider import FakeLexicalRetriever
from backend.app.hybrid_retrieval.fusion import reciprocal_rank_fusion
from backend.app.hybrid_retrieval.lexical_provider import SQLLexicalRetriever
from backend.app.hybrid_retrieval.models import HybridCandidate, HybridRetrievalResponse
from backend.app.hybrid_retrieval.service import HybridRetrievalService

__all__ = [
    "BaseLexicalRetriever",
    "BM25Okapi",
    "FakeLexicalRetriever",
    "HybridCandidate",
    "HybridRetrievalConfig",
    "HybridRetrievalError",
    "HybridRetrievalResponse",
    "HybridRetrievalService",
    "LexicalRetrievalError",
    "SQLLexicalRetriever",
    "reciprocal_rank_fusion",
    "tokenize",
]
