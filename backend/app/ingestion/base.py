"""Base abstractions and data structures for document text extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ExtractedPage:
    """Represents text extracted from a distinct document page or section."""
    page_number: Optional[int]
    text: str
    char_count: int = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class ExtractedDocument:
    """Structured extraction output capturing all pages and document-level telemetry."""
    filename: str
    file_type: str
    pages: List[ExtractedPage]
    total_pages: Optional[int] = None
    total_chars: int = 0
    has_extractable_text: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.total_chars = sum(p.char_count for p in self.pages)
        self.has_extractable_text = any(len(p.text.strip()) > 0 for p in self.pages)
        if self.total_pages is None:
            self.total_pages = len(self.pages) if self.pages else None


class BaseExtractor(ABC):
    """Abstract base class for format-specific text extractors."""

    @abstractmethod
    def extract(self, file_path: Path, filename: str) -> ExtractedDocument:
        """Extracts structured, page-aware text from the target document file."""
        pass
