"""DOCX text extractor using python-docx with paragraph and table extraction."""

from pathlib import Path
from typing import List
import docx
from docx.opc.exceptions import PackageNotFoundError
from backend.app.ingestion.base import BaseExtractor, ExtractedDocument, ExtractedPage


class DOCXExtractor(BaseExtractor):
    """Extracts text from DOCX files preserving paragraph and table structure."""

    def extract(self, file_path: Path, filename: str) -> ExtractedDocument:
        """Extracts text from paragraphs and tabular blocks in logical order."""
        if not file_path.exists():
            raise FileNotFoundError("Document file not found at path.")

        try:
            doc = docx.Document(str(file_path))
        except PackageNotFoundError:
            raise ValueError("Corrupt or invalid DOCX package (missing OpenXML container).")
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX document: {e}")

        extracted_lines: List[str] = []

        # 1. Extract paragraphs
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                extracted_lines.append(text)

        # 2. Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    extracted_lines.append(" | ".join(row_cells))

        full_text = "\n\n".join(extracted_lines)
        pages = [
            ExtractedPage(
                page_number=None,  # DOCX does not have fixed physical page boundaries in OpenXML
                text=full_text,
                metadata={"format": "docx"},
            )
        ]

        return ExtractedDocument(
            filename=filename,
            file_type=".docx",
            pages=pages,
            total_pages=None,
            metadata={"paragraph_count": len(doc.paragraphs), "table_count": len(doc.tables)},
        )
