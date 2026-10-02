"""Local Filesystem Storage Provider Implementation.

Implements BaseStorageProvider using local directory structure with tenant isolation:
storage/uploads/{user_id}/{document_id}/{filename}
"""

import shutil
from pathlib import Path
from typing import BinaryIO, Union
from backend.app.core.config import get_settings
from backend.app.storage.base import BaseStorageProvider


class LocalStorageProvider(BaseStorageProvider):
    """Local filesystem document storage provider."""

    def __init__(self, base_dir: Union[str, Path, None] = None):
        if base_dir is None:
            settings = get_settings()
            self.base_dir = Path(settings.storage.upload_storage_dir)
        else:
            self.base_dir = Path(base_dir)

    def save(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        content: Union[bytes, BinaryIO],
    ) -> str:
        """Saves file to local filesystem under user_id/document_id directory."""
        user_doc_dir = self.base_dir / str(user_id) / str(document_id)
        user_doc_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = user_doc_dir / filename
        if isinstance(content, bytes):
            with open(target_path, "wb") as f:
                f.write(content)
        else:
            with open(target_path, "wb") as f:
                shutil.copyfileobj(content, f)
                
        return str(target_path)

    def read(self, storage_path: str) -> bytes:
        """Reads file bytes from local filesystem path."""
        path = Path(storage_path)
        if not path.exists():
            raise FileNotFoundError(f"Storage file not found: {storage_path}")
        with open(path, "rb") as f:
            return f.read()

    def delete(self, storage_path: str, user_id: str) -> bool:
        """Safely removes the document directory enforcing user isolation check."""
        path = Path(storage_path)
        doc_dir = path.parent
        # Safety check: ensure user_id is in the path hierarchy to prevent arbitrary path traversal deletions
        if doc_dir.exists() and str(user_id) in str(doc_dir):
            shutil.rmtree(doc_dir, ignore_errors=True)
            return True
        return False

    def exists(self, storage_path: str) -> bool:
        """Checks if local file exists."""
        return Path(storage_path).is_file()
