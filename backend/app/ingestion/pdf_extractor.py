"""PDF text extractor leveraging pypdf with page-aware structured extraction."""

from pathlib import Path
from typing import List
import pypdf
from pypdf.errors import PdfReadError
from backend.app.ingestion.base import BaseExtractor, ExtractedDocument, ExtractedPage


class PDFExtractor(BaseExtractor):
    """Extracts text page-by-page from PDF documents."""

    def extract(self, file_path: Path, filename: str) -> ExtractedDocument:
        """Extracts text preserving 1-indexed page boundaries."""
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found at path.")

        try:
            reader = pypdf.PdfReader(str(file_path))
        except PdfReadError as e:
            raise ValueError(f"Corrupt or unreadable PDF document: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read PDF document: {e}")

        if reader.is_encrypted:
            try:
                # Attempt empty password decryption
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    raise ValueError("Encrypted/password-protected PDF is not supported.")
            except Exception:
                raise ValueError("Encrypted/password-protected PDF is not supported.")

        pages: List[ExtractedPage] = []
        total_pages = len(reader.pages)

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
                
            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    text=page_text,
                    metadata={"source_page": page_num},
                )
            )

        has_text = any(len(p.text.strip()) > 0 for p in pages)
        metadata = {
            "page_count": total_pages,
            "is_encrypted": reader.is_encrypted,
            "has_extractable_text": has_text,
            "note": "Scanned or image-only PDF" if not has_text and total_pages > 0 else "Text-based PDF",
        }

        return ExtractedDocument(
            filename=filename,
            file_type=".pdf",
            pages=pages,
            total_pages=total_pages,
            metadata=metadata,
        )
