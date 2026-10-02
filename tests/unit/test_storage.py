"""Unit tests for Document Storage subsystem and LocalStorageProvider."""

import os
import pytest
from pathlib import Path
from backend.app.storage.base import BaseStorageProvider
from backend.app.storage.local import LocalStorageProvider
from backend.app.storage.factory import get_storage_provider


class TestStorageSubsystem:
    """Validates BaseStorageProvider contract, LocalStorageProvider, and factory."""

    def test_factory_returns_storage_provider(self):
        provider = get_storage_provider()
        assert isinstance(provider, BaseStorageProvider)
        assert isinstance(provider, LocalStorageProvider)

    def test_local_storage_lifecycle(self, tmp_path: Path):
        provider = LocalStorageProvider(base_dir=tmp_path)
        user_id = "test-user-123"
        document_id = "doc-uuid-456"
        filename = "document.txt"
        test_content = b"Advanced RAG storage content for testing."

        # 1. Save
        saved_path = provider.save(
            user_id=user_id,
            document_id=document_id,
            filename=filename,
            content=test_content,
        )
        assert os.path.exists(saved_path)
        assert provider.exists(saved_path) is True

        # 2. Read
        read_content = provider.read(saved_path)
        assert read_content == test_content

        # 3. Delete with valid user_id
        delete_result = provider.delete(saved_path, user_id=user_id)
        assert delete_result is True
        assert provider.exists(saved_path) is False
        assert not os.path.exists(saved_path)

    def test_local_storage_delete_isolation_guard(self, tmp_path: Path):
        provider = LocalStorageProvider(base_dir=tmp_path)
        user_id = "test-user-123"
        document_id = "doc-uuid-456"
        filename = "document.txt"
        test_content = b"Secret payload."

        saved_path = provider.save(
            user_id=user_id,
            document_id=document_id,
            filename=filename,
            content=test_content,
        )

        # Deletion attempt with mismatched user_id must be rejected
        cross_tenant_delete = provider.delete(saved_path, user_id="intruder-user-789")
        assert cross_tenant_delete is False
        # File must remain intact
        assert provider.exists(saved_path) is True

    def test_read_nonexistent_file_raises_error(self, tmp_path: Path):
        provider = LocalStorageProvider(base_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            provider.read(str(tmp_path / "nonexistent" / "file.txt"))
