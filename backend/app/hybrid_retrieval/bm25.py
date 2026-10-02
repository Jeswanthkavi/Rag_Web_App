"""Deterministic pure-Python BM25Okapi scoring implementation."""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

# Token pattern matching alpha-numeric words
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase words."""
    if not text:
        return []
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


class BM25Okapi:
    """Deterministic BM25Okapi implementation for lexical corpus ranking."""

    def __init__(
        self,
        corpus: Sequence[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size > 0 else 0.0
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._initialize(corpus)

    def _initialize(self, corpus: Sequence[List[str]]) -> None:
        """Builds document frequency table and inverted index."""
        df: Dict[str, int] = defaultdict(int)
        for doc_id, doc in enumerate(corpus):
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            for token, count in freq.items():
                df[token] += 1
                self.inverted_index[token].append((doc_id, count))

        # Calculate IDF values
        # Formula: ln(1 + (N - n + 0.5) / (n + 0.5))
        for token, freq in df.items():
            idf_val = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[token] = idf_val

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Calculates BM25 relevance scores for all documents in the corpus for the given query."""
        scores = [0.0] * self.corpus_size
        if not query_tokens or self.corpus_size == 0 or self.avgdl == 0.0:
            return scores

        # Calculate score contributions per query term
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            postings = self.inverted_index[token]

            for doc_id, tf in postings:
                dl = self.doc_len[doc_id]
                # BM25 numerator and denominator
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf_val * (num / denom)

        return scores
