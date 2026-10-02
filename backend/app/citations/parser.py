"""Citation parsing and verification against authoritative retrieved chunks."""

import re
import logging
from typing import Any, List, Sequence, Set, Union
from backend.app.citations.models import Citation
from backend.app.reranking.models import RerankResult
from backend.app.retrieval.models import RetrievalResult

logger = logging.getLogger("app.citations.parser")

# Regex to capture [Source 1], [Source: 1], [source 1], or [1] citation markers
CITATION_REGEX = re.compile(r"\[(?:Source\s*:?\s*)?(\d+)\]", re.IGNORECASE)


class CitationParser:
    """Extracts, validates, and maps citation references in LLM-generated answers."""

    def extract_source_indices(self, text: str) -> List[int]:
        """Extracts unique source index integers from text in chronological order of appearance."""
        if not text:
            return []

        matches = CITATION_REGEX.findall(text)
        seen: Set[int] = set()
        ordered_indices: List[int] = []

        for match in matches:
            try:
                idx = int(match)
                if idx not in seen:
                    seen.add(idx)
                    ordered_indices.append(idx)
            except (ValueError, TypeError):
                continue

        return ordered_indices

    def parse_and_verify(
        self,
        answer_text: str,
        retrieved_results: Sequence[Union[RetrievalResult, RerankResult, Any]],
    ) -> List[Citation]:
        """Parses citation markers from the answer and verifies them against retrieved context.
        
        Rules:
        1. Only source indices between 1 and len(retrieved_results) are accepted.
        2. Any out-of-bounds or fabricated citation indices are rejected.
        3. All citation metadata (document_id, chunk_id, filename, page_number, chunk_index)
           is populated strictly from the authoritative RetrievalResult objects.
        4. Duplicates are filtered while preserving appearance order.
        """
        if not answer_text or not retrieved_results:
            return []

        total_retrieved = len(retrieved_results)
        source_indices = self.extract_source_indices(answer_text)
        valid_citations: List[Citation] = []

        for source_num in source_indices:
            # 1-indexed verification
            if 1 <= source_num <= total_retrieved:
                matched_result: RetrievalResult = retrieved_results[source_num - 1]
                citation = Citation(
                    citation_id=f"source_{source_num}",
                    document_id=matched_result.document_id,
                    chunk_id=matched_result.chunk_id,
                    filename=matched_result.filename,
                    page_number=matched_result.page_number,
                    chunk_index=matched_result.chunk_index,
                )
                valid_citations.append(citation)
            else:
                logger.warning(
                    f"Rejected hallucinated/out-of-bounds citation reference [Source {source_num}] (total retrieved: {total_retrieved})"
                )

        return valid_citations
