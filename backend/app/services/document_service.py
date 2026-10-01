"""Document Service orchestrating secure document validation, storage, and lifecycle operations."""

import os
import shutil
import uuid
from pathlib import Path
from typing import Tuple, List, Optional
from fastapi import UploadFile, HTTPException, status
from backend.app.core.config import get_settings
from backend.app.models.user import User
from backend.app.models.document import Document, DocumentStatus
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.utils.file_validation import (
    sanitize_filename,
    validate_file_extension,
    validate_file_header_content,
    validate_file_size,
    MAX_FILE_SIZE_BYTES,
)

settings = get_settings()


class DocumentService:
    """Service encapsulating secure multi-tenant document management workflows."""

    def __init__(self, doc_repo: DocumentRepository):
        self.doc_repo = doc_repo

    async def upload_document(self, user: User, file: UploadFile) -> Document:
        """Processes, validates, stores, and catalogs a new document for the authenticated user."""
        raw_filename = file.filename or "uploaded_document.txt"
        sanitized_filename = sanitize_filename(raw_filename)

        # 1. Validate extension
        try:
            file_type = validate_file_extension(sanitized_filename)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        # 2. Setup isolated user storage path
        document_id = str(uuid.uuid4())
        base_storage = Path(settings.storage.upload_storage_dir)
        user_doc_dir = base_storage / str(user.id) / document_id
        user_doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Physical filename is strictly system-managed to prevent filesystem exploitation
        physical_storage_path = user_doc_dir / f"document{file_type}"

        # 3. Stream and write file safely to disk with continuous size and header verification
        total_bytes = 0
        chunk_size = 64 * 1024  # 64 KB
        first_chunk = True

        try:
            with open(physical_storage_path, "wb") as buffer:
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    
                    total_bytes += len(chunk)
                    
                    # Validate header signature on first chunk
                    if first_chunk:
                        validate_file_header_content(chunk, file_type)
                        first_chunk = False
                        
                    # Check maximum size limits
                    if total_bytes > MAX_FILE_SIZE_BYTES:
                        raise ValueError(f"File size exceeds maximum allowed limit of 25 MB.")
                        
                    buffer.write(chunk)
                    
            # Check non-empty
            validate_file_size(total_bytes, MAX_FILE_SIZE_BYTES)
            
        except ValueError as e:
            # Clean up filesystem artifacts on validation failure
            if user_doc_dir.exists():
                shutil.rmtree(user_doc_dir, ignore_errors=True)
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if "exceeds" in str(e).lower() else status.HTTP_400_BAD_REQUEST
            raise HTTPException(
                status_code=status_code,
                detail=str(e),
            )
        except Exception as e:
            # Clean up filesystem artifacts on unexpected error
            if user_doc_dir.exists():
                shutil.rmtree(user_doc_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during file upload processing.",
            )

        # 4. Insert metadata record in database
        try:
            document = self.doc_repo.create(
                user_id=user.id,
                filename=sanitized_filename,
                file_type=file_type,
                file_size_bytes=total_bytes,
                storage_path=str(physical_storage_path),
                status=DocumentStatus.PENDING,
                id=document_id,
            )
            return document
        except Exception:
            # Rollback filesystem write if database cataloging fails
            if user_doc_dir.exists():
                shutil.rmtree(user_doc_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist document record in database.",
            )

    def list_documents(
        self,
        user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> Tuple[List[Document], int]:
        """Lists metadata for documents owned strictly by the authenticated user."""
        return self.doc_repo.list_by_user(user_id=user.id, skip=skip, limit=limit)

    def get_document(self, user: User, document_id: str) -> Document:
        """Retrieves a single document verifying strict user ownership."""
        doc = self.doc_repo.get_by_id(document_id=document_id, user_id=user.id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )
        return doc

    def delete_document(self, user: User, document_id: str) -> None:
        """Deletes a document and purges its physical storage directory."""
        doc = self.get_document(user=user, document_id=document_id)
        
        # 1. Remove database record
        self.doc_repo.delete(doc)
        
        # 2. Clean up filesystem storage directory safely
        storage_path = Path(doc.storage_path)
        doc_dir = storage_path.parent
        if doc_dir.exists() and str(user.id) in str(doc_dir):
            shutil.rmtree(doc_dir, ignore_errors=True)
