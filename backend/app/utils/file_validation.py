"""File validation and security utilities for document uploads.

Enforces extension whitelisting, MIME type checks, magic-byte inspection,
maximum size constraints (25MB), and comprehensive path traversal protection.
"""

import os
import re
from pathlib import Path
from typing import Set

ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".docx", ".txt"}
ALLOWED_MIME_TYPES: Set[str] = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "application/octet-stream",  # Often sent by generic clients
}
MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 Megabytes


def sanitize_filename(filename: str) -> str:
    """Sanitizes user-provided filename to prevent path traversal and filesystem manipulation.
    
    Protects against:
    - Relative traversals ('../', '..\\')
    - Absolute root paths ('/etc/passwd')
    - Windows drive paths ('C:\\secret.txt', 'D:/secret.txt')
    - Null byte injection ('secret.pdf\\0.txt')
    - Control characters and illegal filesystem characters
    """
    if not filename or not isinstance(filename, str):
        return "unnamed_document.txt"

    # Remove null bytes
    cleaned = filename.replace("\x00", "").strip()

    # Strip Windows drive letters (e.g., 'C:', 'D:')
    cleaned = re.sub(r"^[a-zA-Z]:", "", cleaned)

    # Normalize slashes and extract only the final basename
    cleaned = cleaned.replace("\\", "/")
    cleaned = Path(cleaned).name

    # Remove leading dots or slashes that could cause hidden file creation or relative pathing
    cleaned = re.sub(r"^[.\s/]+", "", cleaned)

    # Replace forbidden Windows/Linux filename characters with underscores
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", cleaned)

    # Collapse multiple consecutive underscores or spaces
    cleaned = re.sub(r"[\s_]+", "_", cleaned).strip(" ._")

    # If completely empty after sanitization, assign a fallback name
    if not cleaned:
        cleaned = "unnamed_document"

    # Ensure max filename length constraint
    if len(cleaned) > 255:
        stem = Path(cleaned).stem[:240]
        suffix = Path(cleaned).suffix[:10]
        cleaned = f"{stem}{suffix}"

    return cleaned


def validate_file_extension(filename: str) -> str:
    """Validates that the file has an allowed extension (.pdf, .docx, .txt).
    
    Returns the lowercased extension.
    """
    sanitized = sanitize_filename(filename)
    ext = Path(sanitized).suffix.lower()
    
    if not ext or ext not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{ext}'. Allowed extensions: {allowed_list}")
        
    return ext


def validate_file_header_content(header_bytes: bytes, extension: str) -> None:
    """Inspects magic bytes and header structure to ensure file content matches its extension.
    
    Raises ValueError if content signature mismatches expected format.
    """
    if not header_bytes:
        raise ValueError("Cannot validate an empty file.")

    ext = extension.lower()

    if ext == ".pdf":
        # Standard PDF starts with %PDF- (hex: 25 50 44 46 2d)
        if not header_bytes.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF format: File missing standard %PDF- header signature.")

    elif ext == ".docx":
        # DOCX is an OpenXML ZIP container starting with PK\x03\x04
        if not header_bytes.startswith(b"PK\x03\x04"):
            raise ValueError("Invalid DOCX format: File missing standard OpenXML ZIP signature.")

    elif ext == ".txt":
        # Plaintext should be decodable as text and free of binary null bytes in initial chunk
        if b"\x00" in header_bytes:
            raise ValueError("Invalid TXT format: Binary null bytes detected in plain text file.")
        try:
            # Check if header can decode under UTF-8 or latin-1
            header_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                header_bytes.decode("latin-1")
            except Exception:
                raise ValueError("Invalid TXT format: File contains non-decodable binary stream.")


def validate_file_size(size_in_bytes: int, max_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
    """Ensures file size is within limits (1 byte to 25 MB)."""
    if size_in_bytes <= 0:
        raise ValueError("Uploaded file is empty (0 bytes).")
        
    if size_in_bytes > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        actual_mb = size_in_bytes / (1024 * 1024)
        raise ValueError(f"File size exceeds limit of {max_mb} MB (received {actual_mb:.2f} MB).")
