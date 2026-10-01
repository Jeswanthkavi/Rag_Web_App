"""Users endpoints for profile management."""

from fastapi import APIRouter, Depends, status
from backend.app.api.dependencies import get_current_active_user
from backend.app.models.user import User
from backend.app.schemas.user import UserResponse

router = APIRouter()


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current user profile",
    description="Retrieves profile details for the authenticated, active user. Never exposes password hashes.",
)
def get_me(
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """Returns the authenticated user profile."""
    return UserResponse.model_validate(current_user)
