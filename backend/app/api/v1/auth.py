"""Authentication endpoints (Registration and Login)."""

from fastapi import APIRouter, Depends, status
from backend.app.api.dependencies import get_auth_service
from backend.app.services.auth_service import AuthService
from backend.app.schemas.user import UserCreate, UserResponse
from backend.app.schemas.auth import LoginRequest, Token

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description="Registers a new tenant user with email and password. Does not return password hashes.",
)
def register(
    user_in: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """Handles new user registration."""
    user = auth_service.register_user(user_in)
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="User login and JWT token issuance",
    description="Authenticates credentials and returns a signed JWT access token.",
)
def login(
    credentials: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> Token:
    """Handles user login and issues JWT token."""
    user = auth_service.authenticate_user(
        email=credentials.email,
        password=credentials.password,
    )
    return auth_service.create_user_token(user)
