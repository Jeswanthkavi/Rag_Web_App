"""Semantic Retrieval & Similarity Search API Endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.api.dependencies import get_current_active_user, get_retrieval_service
from backend.app.models.user import User
from backend.app.retrieval.models import (
    RetrievalError,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalValidationError,
)
from backend.app.retrieval.service import RetrievalService

logger = logging.getLogger("app.api.v1.retrieval")

router = APIRouter()


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic similarity search across document chunks",
    description="Embeds the natural language query, searches Qdrant using cosine similarity, applies strict tenant isolation, and returns ranked chunk results with citation metadata.",
)
def semantic_search(
    request: RetrievalQuery,
    current_user: User = Depends(get_current_active_user),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    """Executes tenant-isolated semantic retrieval."""
    try:
        response = retrieval_service.search(
            query=request.query,
            user_id=str(current_user.id),
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_id=request.document_id,
        )
        return response

    except RetrievalValidationError as e:
        logger.warning(f"Retrieval validation error for user '{current_user.id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    except RetrievalError as e:
        logger.error(f"Retrieval error for user '{current_user.id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Semantic retrieval failed due to an internal error.",
        ) from e

    except Exception as e:
        logger.error(f"Unexpected retrieval error for user '{current_user.id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during semantic search.",
        ) from e
