"""Grounded RAG Answer Generation API Endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.api.dependencies import get_current_active_user, get_rag_service
from backend.app.models.user import User
from backend.app.rag.models import (
    RAGError,
    RAGQuery,
    RAGResponse,
    RAGValidationError,
)
from backend.app.rag.service import RAGService

logger = logging.getLogger("app.api.v1.rag")

router = APIRouter()


@router.post(
    "/query",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute grounded RAG answer generation",
    description="Retrieves relevant tenant document chunks via semantic search, builds a grounded prompt, and synthesizes a factual answer using OpenRouter / NVIDIA Nemotron.",
)
def rag_query(
    request: RAGQuery,
    current_user: User = Depends(get_current_active_user),
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """Executes grounded RAG answer generation for the authenticated user."""
    try:
        response = rag_service.generate_answer(
            query=request.query,
            user_id=str(current_user.id),
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_id=request.document_id,
            conversation_id=request.conversation_id,
        )
        return response

    except RAGValidationError as e:
        logger.warning(f"RAG validation error for user '{current_user.id}': {e}")
        err_msg = str(e)
        if "Conversation" in err_msg and "not found" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=err_msg,
            ) from e
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=err_msg,
        ) from e

    except RAGError as e:
        logger.error(f"RAG generation error for user '{current_user.id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Grounded answer generation failed due to an upstream provider error.",
        ) from e

    except Exception as e:
        logger.error(f"Unexpected RAG error for user '{current_user.id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during RAG query execution.",
        ) from e
