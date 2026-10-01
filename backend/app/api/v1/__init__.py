"""API v1 Router aggregation."""

from fastapi import APIRouter
from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.users import router as users_router

api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users_router, prefix="/users", tags=["Users"])

__all__ = ["api_router"]
