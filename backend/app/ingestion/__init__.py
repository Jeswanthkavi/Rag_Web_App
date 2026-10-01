"""Ingestion Package for document parsing, extraction, and normalization."""

from backend.app.ingestion.base import BaseExtractor, ExtractedDocument, ExtractedPage
from backend.app.ingestion.pdf_extractor import PDFExtractor
from backend.app.ingestion.docx_extractor import DOCXExtractor
from backend.app.ingestion.txt_extractor import TXTExtractor
from backend.app.ingestion.normalizer import TextNormalizer
from backend.app.ingestion.service import IngestionService, run_document_ingestion

__all__ = [
    "BaseExtractor",
    "ExtractedDocument",
    "ExtractedPage",
    "PDFExtractor",
    "DOCXExtractor",
    "TXTExtractor",
    "TextNormalizer",
    "IngestionService",
    "run_document_ingestion",
]
