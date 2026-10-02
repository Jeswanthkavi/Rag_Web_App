"""Deterministic evaluation metrics for answer relevance and faithfulness/groundedness."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Set
from backend.app.evaluation.models import GenerationMetrics

WORD_REGEX = re.compile(r"\b\w+\b")


def _extract_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    # Exclude very short stopwords
    stopwords = {"the", "a", "an", "in", "on", "at", "of", "to", "for", "is", "are", "was", "were", "and", "or"}
    tokens = {t.lower() for t in WORD_REGEX.findall(text)}
    return tokens - stopwords


def compute_answer_relevance(generated_answer: str, expected_answer: str, query: str = "") -> float:
    """Computes answer relevance based on token overlap (Jaccard similarity) against expected ground truth.
    
    Args:
        generated_answer: The synthesized LLM answer text.
        expected_answer: Ground truth target answer.
        query: Optional question string.
        
    Returns:
        Float score [0.0 - 1.0].
    """
    if not generated_answer or not expected_answer:
        return 0.0

    gen_tokens = _extract_tokens(generated_answer)
    exp_tokens = _extract_tokens(expected_answer)

    if not exp_tokens or not gen_tokens:
        return 0.0

    intersection = gen_tokens.intersection(exp_tokens)
    union = gen_tokens.union(exp_tokens)
    return float(len(intersection)) / float(len(union)) if union else 0.0


def compute_faithfulness(generated_answer: str, retrieved_contexts: Sequence[str]) -> float:
    """Computes faithfulness / groundedness: proportion of answer content tokens grounded in retrieved context.
    
    Args:
        generated_answer: Synthesized answer string.
        retrieved_contexts: List of chunk texts retrieved from documents.
        
    Returns:
        Float score [0.0 - 1.0].
    """
    if not generated_answer or not retrieved_contexts:
        return 0.0

    gen_tokens = _extract_tokens(generated_answer)
    if not gen_tokens:
        return 1.0

    # Aggregate context tokens
    context_tokens: Set[str] = set()
    for ctx in retrieved_contexts:
        context_tokens.update(_extract_tokens(ctx))

    if not context_tokens:
        return 0.0

    supported = gen_tokens.intersection(context_tokens)
    return float(len(supported)) / float(len(gen_tokens))


class BaseJudgeProvider(ABC):
    """Abstract interface for LLM / heuristic judges."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        generated_answer: str,
        expected_answer: Optional[str],
        retrieved_contexts: Sequence[str],
    ) -> tuple[float, float]:
        """Returns (answer_relevance, faithfulness)."""
        pass


class DeterministicJudge(BaseJudgeProvider):
    """Deterministic token-based judge requiring no external API calls."""

    def evaluate_answer_relevance(self, query: str, answer: str, expected_answer: Optional[str] = None) -> float:
        return compute_answer_relevance(answer, expected_answer or "", query)

    def evaluate_faithfulness(self, answer: str, context_chunks: Sequence[str]) -> float:
        return compute_faithfulness(answer, context_chunks)

    def evaluate(
        self,
        query: str,
        generated_answer: str,
        expected_answer: Optional[str],
        retrieved_contexts: Sequence[str],
    ) -> tuple[float, float]:
        rel = compute_answer_relevance(generated_answer, expected_answer or "", query) if expected_answer else 1.0
        faith = compute_faithfulness(generated_answer, retrieved_contexts)
        return rel, faith


class FakeJudge(BaseJudgeProvider):
    """Mock judge for testing custom metric thresholds."""

    def __init__(
        self,
        answer_relevance: float = 1.0,
        faithfulness: float = 1.0,
        fixed_relevance: Optional[float] = None,
        fixed_faithfulness: Optional[float] = None,
    ):
        self._answer_relevance = fixed_relevance if fixed_relevance is not None else answer_relevance
        self._faithfulness = fixed_faithfulness if fixed_faithfulness is not None else faithfulness

    def evaluate_answer_relevance(self, query: str, answer: str, expected_answer: Optional[str] = None) -> float:
        return self._answer_relevance

    def evaluate_faithfulness(self, answer: str, context_chunks: Sequence[str]) -> float:
        return self._faithfulness

    def evaluate(
        self,
        query: str,
        generated_answer: str,
        expected_answer: Optional[str],
        retrieved_contexts: Sequence[str],
    ) -> tuple[float, float]:
        return self._answer_relevance, self._faithfulness

