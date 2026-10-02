"""Context construction component for formatting retrieved chunks into structured prompt context."""

from typing import List
from backend.app.retrieval.models import RetrievalResult


class ContextBuilder:
    """Formats Phase 11 RetrievalResult objects into deterministic grounded context text."""

    def build_context(self, results: List[RetrievalResult]) -> str:
        """Constructs formatted context text with explicit document boundaries.
        
        Preserves metadata (filename, page number, chunk index) for future citation rendering
        while presenting a clean, isolated reference structure for the LLM.
        """
        if not results:
            return ""

        context_blocks: List[str] = []
        for idx, item in enumerate(results, start=1):
            page_info = f" (Page {item.page_number})" if item.page_number is not None else ""
            header = f"[Context {idx}]\nDocument: {item.filename}{page_info}"
            body = f"Content:\n{item.chunk_text.strip()}"
            context_blocks.append(f"{header}\n{body}")

        return "\n\n".join(context_blocks)
