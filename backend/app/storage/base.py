"""Abstract Base Class for Document Storage Providers.

Defines the contract for persisting, reading, verifying, and deleting uploaded files
across Local Filesystem and Cloud Object Storage (e.g., S3, GCS, Azure Blob, Cloudflare R2).
"""

from abc import ABC, abstractmethod
from typing import BinaryIO, Union


class BaseStorageProvider(ABC):
    """Abstract interface for document file storage."""

    @abstractmethod
    def save(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        content: Union[bytes, BinaryIO],
    ) -> str:
        """Persists file content and returns a canonical storage identifier or file path.

        Args:
            user_id: Unique user identifier for tenant isolation.
            document_id: Unique document UUID.
            filename: Target file name (including extension).
            content: Raw file bytes or file-like binary stream.

        Returns:
            Canonical storage path or URI string.
        """
        pass

    @abstractmethod
    def read(self, storage_path: str) -> bytes:
        """Reads and returns the complete binary content of the stored file.

        Args:
            storage_path: Canonical storage path or URI.

        Returns:
            Raw bytes of the document.
        """
        pass

    @abstractmethod
    def delete(self, storage_path: str, user_id: str) -> bool:
        """Deletes the stored file and its enclosing document directory/prefix.

        Args:
            storage_path: Canonical storage path or URI.
            user_id: User identifier to enforce tenant isolation.

        Returns:
            True if deletion was successful or file did not exist, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, storage_path: str) -> bool:
        """Checks whether the file exists at the given storage location.

        Args:
            storage_path: Canonical storage path or URI.

        Returns:
            True if exists, False otherwise.
        """
        pass
