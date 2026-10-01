"""Unit tests for file validation and filename sanitization (Phase 6)."""

import pytest
from backend.app.utils.file_validation import (
    sanitize_filename,
    validate_file_extension,
    validate_file_header_content,
    validate_file_size,
    MAX_FILE_SIZE_BYTES,
)


class TestFilenameSanitization:
    """Verify filename sanitization protects against path traversal and forbidden characters."""

    def test_sanitize_standard_filenames(self):
        assert sanitize_filename("annual_report_2026.pdf") == "annual_report_2026.pdf"
        assert sanitize_filename("my document.docx") == "my_document.docx"

    def test_sanitize_path_traversals(self):
        assert sanitize_filename("../../secret.txt") == "secret.txt"
        assert sanitize_filename("..\\..\\secret.txt") == "secret.txt"
        assert sanitize_filename("../../../etc/passwd") == "passwd"

    def test_sanitize_windows_drive_paths(self):
        assert sanitize_filename("C:\\Users\\Admin\\secret.pdf") == "secret.pdf"
        assert sanitize_filename("D:/Data/Documents/report.docx") == "report.docx"

    def test_sanitize_null_bytes(self):
        assert sanitize_filename("document.pdf\x00.exe") == "document.pdf.exe"

    def test_sanitize_empty_or_forbidden_characters(self):
        assert sanitize_filename('test:file*name?.txt') == "test_file_name_.txt"
        assert sanitize_filename("...") == "unnamed_document"
        assert sanitize_filename("") == "unnamed_document.txt"


class TestFileExtensionValidation:
    """Verify supported extensions are accepted and others rejected."""

    def test_allowed_extensions(self):
        assert validate_file_extension("paper.pdf") == ".pdf"
        assert validate_file_extension("notes.DOCX") == ".docx"
        assert validate_file_extension("data.txt") == ".txt"

    def test_unsupported_extensions(self):
        with pytest.raises(ValueError) as exc:
            validate_file_extension("script.py")
        assert "Unsupported file extension" in str(exc.value)

        with pytest.raises(ValueError):
            validate_file_extension("executable.exe")

        with pytest.raises(ValueError):
            validate_file_extension("archive.tar.gz")


class TestFileHeaderValidation:
    """Verify magic bytes match file extensions."""

    def test_valid_pdf_header(self):
        pdf_bytes = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        validate_file_header_content(pdf_bytes, ".pdf")  # Should not raise

    def test_invalid_pdf_header_raises_error(self):
        with pytest.raises(ValueError) as exc:
            validate_file_header_content(b"NOT A VALID PDF HEADER", ".pdf")
        assert "missing standard %PDF-" in str(exc.value)

    def test_valid_docx_header(self):
        docx_bytes = b"PK\x03\x04\x14\x00\x06\x00"
        validate_file_header_content(docx_bytes, ".docx")  # Should not raise

    def test_invalid_docx_header_raises_error(self):
        with pytest.raises(ValueError) as exc:
            validate_file_header_content(b"NOT A ZIP DOCX CONTAINER", ".docx")
        assert "missing standard OpenXML ZIP" in str(exc.value)

    def test_valid_txt_header(self):
        txt_bytes = "This is a valid UTF-8 plain text file.".encode("utf-8")
        validate_file_header_content(txt_bytes, ".txt")  # Should not raise

    def test_invalid_txt_binary_null_bytes_raises_error(self):
        binary_txt = b"Text with \x00 null bytes."
        with pytest.raises(ValueError) as exc:
            validate_file_header_content(binary_txt, ".txt")
        assert "null bytes detected" in str(exc.value)


class TestFileSizeValidation:
    """Verify size boundaries."""

    def test_valid_file_size(self):
        validate_file_size(1024)
        validate_file_size(MAX_FILE_SIZE_BYTES)

    def test_zero_byte_file_raises_error(self):
        with pytest.raises(ValueError) as exc:
            validate_file_size(0)
        assert "empty" in str(exc.value)

    def test_oversized_file_raises_error(self):
        with pytest.raises(ValueError) as exc:
            validate_file_size(MAX_FILE_SIZE_BYTES + 1)
        assert "exceeds limit" in str(exc.value)
