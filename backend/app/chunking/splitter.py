"""Deterministic text splitter with hierarchical boundary awareness and configurable overlap."""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from backend.app.ingestion.base import ExtractedPage


@dataclass
class ChunkData:
    """In-memory representation of an individual text chunk with metadata."""
    chunk_index: int
    text: str
    page_number: Optional[int]
    token_count: int
    char_count: int

    def __post_init__(self):
        self.char_count = len(self.text)


def count_tokens_approx(text: str) -> int:
    """Approximates token count using deterministic whitespace tokenization.
    
    Phase 8 baseline strategy: Counts non-whitespace token sequences.
    Phase 9 will introduce exact embedding model tokenizers.
    """
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


class RecursiveCharacterSplitter:
    """Hierarchical text splitter respecting semantic, sentence, and word boundaries."""

    DEFAULT_SEPARATORS: List[str] = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        "? ",    # Question sentences
        "! ",    # Exclamation sentences
        "; ",    # Semicolons
        " ",     # Words
        "",      # Characters (fallback)
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        if separator:
            return text.split(separator)
        return list(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merges small splits into chunks respecting chunk_size and chunk_overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0
        sep_len = len(separator) if separator else 0

        for s in splits:
            s_len = len(s)
            if not s.strip():
                continue

            # If current split itself exceeds chunk_size, we recurse on lower separators
            if total_len + s_len + (sep_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc_text = separator.join(current_doc).strip()
                    if doc_text:
                        docs.append(doc_text)
                    
                    # Handle overlap: keep suffix of current_doc up to chunk_overlap length
                    while current_doc and total_len > self.chunk_overlap:
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + (sep_len if current_doc else 0)
                        
                current_doc.append(s)
                total_len = sum(len(x) for x in current_doc) + (sep_len * (len(current_doc) - 1))
            else:
                current_doc.append(s)
                total_len += s_len + (sep_len if len(current_doc) > 1 else 0)

        if current_doc:
            doc_text = separator.join(current_doc).strip()
            if doc_text and (not docs or docs[-1] != doc_text):
                docs.append(doc_text)

        return docs

    def split_text(self, text: str) -> List[str]:
        """Recursively splits a string into chunks within chunk_size limits."""
        text = text.strip()
        if not text:
            return []

        # If text already fits within chunk_size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]

        # Find the highest-priority separator present in text
        chosen_sep = ""
        for sep in self.separators:
            if sep == "" or sep in text:
                chosen_sep = sep
                break

        splits = self._split_text_with_separator(text, chosen_sep)
        
        # Merge sub-splits up to chunk_size
        final_chunks: List[str] = []
        accumulated_splits: List[str] = []

        for s in splits:
            if len(s) > self.chunk_size and chosen_sep != "":
                # Flush accumulated splits before recursing
                if accumulated_splits:
                    merged = self._merge_splits(accumulated_splits, chosen_sep)
                    final_chunks.extend(merged)
                    accumulated_splits = []
                
                # Next lower separator list
                next_seps = self.separators[self.separators.index(chosen_sep) + 1 :]
                sub_splitter = RecursiveCharacterSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=next_seps,
                )
                final_chunks.extend(sub_splitter.split_text(s))
            else:
                accumulated_splits.append(s)

        if accumulated_splits:
            merged = self._merge_splits(accumulated_splits, chosen_sep)
            final_chunks.extend(merged)

        # Filter out empty or whitespace chunks
        return [c.strip() for c in final_chunks if c.strip()]

    def split_document_pages(self, pages: List["ExtractedPage"]) -> List[ChunkData]:
        """Splits page-aware extracted document pages into sequential ChunkData items.
        
        Preserves page boundaries locally for precision citations (e.g. PDF page 1, 2...).
        """
        chunks: List[ChunkData] = []
        global_index = 0

        for page in pages:
            page_text = page.text.strip() if page.text else ""
            if not page_text:
                continue

            page_chunks = self.split_text(page_text)
            for chunk_str in page_chunks:
                clean_chunk = chunk_str.strip()
                if not clean_chunk:
                    continue

                chunks.append(
                    ChunkData(
                        chunk_index=global_index,
                        text=clean_chunk,
                        page_number=page.page_number,
                        token_count=count_tokens_approx(clean_chunk),
                        char_count=len(clean_chunk),
                    )
                )
                global_index += 1

        return chunks
