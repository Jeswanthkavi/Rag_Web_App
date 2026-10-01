"""Document Management API Endpoints (Upload, List, Get, Delete)."""

from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, Query, status
from backend.app.api.dependencies import get_current_active_user, get_document_service
from backend.app.models.user import User
from backend.app.services.document_service import DocumentService
from backend.app.schemas.document import DocumentResponse, DocumentListResponse

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new document",
    description="Uploads a PDF, DOCX, or TXT document (max 25MB). Validates content, stores safely, and catalogs metadata as PENDING.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (.pdf, .docx, .txt)"),
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Handles secure file upload with tenancy isolation."""
    doc = await doc_service.upload_document(user=current_user, file=file)
    return DocumentResponse.model_validate(doc)


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List current user's documents",
    description="Retrieves a paginated list of document metadata records owned strictly by the authenticated user.",
)
def list_documents(
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=500, description="Pagination limit"),
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """Lists documents owned by the active tenant."""
    items, total = doc_service.list_documents(user=current_user, skip=skip, limit=limit)
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(item) for item in items],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document details",
    description="Retrieves metadata for a specific document owned by the authenticated user.",
)
def get_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Retrieves document metadata strictly verifying ownership."""
    doc = doc_service.get_document(user=current_user, document_id=document_id)
    return DocumentResponse.model_validate(doc)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Deletes a document record and purges its physical storage. Enforces strict user ownership.",
)
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    doc_service: DocumentService = Depends(get_document_service),
) -> None:
    """Deletes document and removes filesystem storage."""
    doc_service.delete_document(user=current_user, document_id=document_id)
