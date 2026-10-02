"""Conversations and Messages API Endpoints."""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_current_active_user, get_conversation_service
from backend.app.models.user import User
from backend.app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationDetailResponse,
)
from backend.app.services.conversation_service import ConversationService

logger = logging.getLogger("app.api.v1.conversations")

router = APIRouter()


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session",
)
def create_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    conv_service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Creates a new conversation thread owned by the authenticated user."""
    conv = conv_service.create_conversation(
        user_id=str(current_user.id),
        title=request.title or "New Conversation",
    )
    return ConversationResponse.model_validate(conv)


@router.get(
    "",
    response_model=List[ConversationResponse],
    status_code=status.HTTP_200_OK,
    summary="List all user conversation sessions",
)
def list_conversations(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    conv_service: ConversationService = Depends(get_conversation_service),
) -> List[ConversationResponse]:
    """Lists conversations belonging strictly to the authenticated user."""
    convs = conv_service.list_conversations(
        user_id=str(current_user.id),
        skip=skip,
        limit=limit,
    )
    return [ConversationResponse.model_validate(c) for c in convs]


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get conversation detail and message history",
)
def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    conv_service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetailResponse:
    """Retrieves a conversation and its messages. Returns 404 if not found or unauthorized."""
    conv = conv_service.get_conversation_with_messages(
        conversation_id=conversation_id,
        user_id=str(current_user.id),
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return ConversationDetailResponse.model_validate(conv)


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="Update conversation title",
)
def update_conversation(
    conversation_id: str,
    request: ConversationUpdate,
    current_user: User = Depends(get_current_active_user),
    conv_service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Updates a conversation title."""
    conv = conv_service.update_title(
        conversation_id=conversation_id,
        user_id=str(current_user.id),
        title=request.title,
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return ConversationResponse.model_validate(conv)


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation and all its messages",
)
def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    conv_service: ConversationService = Depends(get_conversation_service),
):
    """Deletes a conversation. Returns 404 if not found or unauthorized."""
    success = conv_service.delete_conversation(
        conversation_id=conversation_id,
        user_id=str(current_user.id),
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return None
