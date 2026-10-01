"""TXT plain text extractor supporting UTF-8 and standard text streams."""

from pathlib import Path
from backend.app.ingestion.base import BaseExtractor, ExtractedDocument, ExtractedPage


class TXTExtractor(BaseExtractor):
    """Extracts text from plain text files."""

    def extract(self, file_path: Path, filename: str) -> ExtractedDocument:
        """Reads and extracts plain text content with UTF-8 decoding."""
        if not file_path.exists():
            raise FileNotFoundError("Document file not found at path.")

        content_bytes = file_path.read_bytes()
        
        # Decode as UTF-8 with fallback to Latin-1
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Failed to decode text file with supported encodings (UTF-8, Latin-1): {e}")

        pages = [
            ExtractedPage(
                page_number=None,  # Plaintext files do not contain physical page divisions
                text=text,
                metadata={"encoding": "utf-8"},
            )
        ]

        return ExtractedDocument(
            filename=filename,
            file_type=".txt",
            pages=pages,
            total_pages=None,
            metadata={"byte_size": len(content_bytes)},
        )
