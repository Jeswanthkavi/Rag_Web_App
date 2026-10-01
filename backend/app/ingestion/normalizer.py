"""Deterministic text normalization for raw extracted document text.

Enforces Unicode NFKC normalization, line-ending standardization, control-character removal,
and whitespace cleanup while strictly preserving paragraph structures and semantic integrity.
"""

import re
import unicodedata


class TextNormalizer:
    """Normalizes raw text extracted from documents into clean, consistent text."""

    @staticmethod
    def normalize(text: str) -> str:
        """Applies comprehensive text normalization pipeline."""
        if not text or not isinstance(text, str):
            return ""

        # 1. Unicode NFKC Canonical Normalization
        normalized = unicodedata.normalize("NFKC", text)

        # 2. Standardize line endings to Unix style (\n)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Strip non-printable control characters (preserve standard tab \t and newline \n)
        normalized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", normalized)

        # 4. Normalize spaces within individual lines (replace tabs and multiple spaces with a single space)
        lines = []
        for line in normalized.split("\n"):
            # Replace horizontal tabs and multiple spaces with single space
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(cleaned_line)

        # 5. Reassemble and normalize consecutive blank lines (limit to maximum 2 consecutive newlines)
        reassembled = "\n".join(lines)
        reassembled = re.sub(r"\n{3,}", "\n\n", reassembled)

        return reassembled.strip()
