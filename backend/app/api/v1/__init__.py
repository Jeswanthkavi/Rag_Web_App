"""API v1 Router aggregation."""

from fastapi import APIRouter
from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.users import router as users_router
from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.retrieval import router as retrieval_router
from backend.app.api.v1.rag import router as rag_router
from backend.app.api.v1.conversations import router as conversations_router
from backend.app.api.v1.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users_router, prefix="/users", tags=["Users"])
api_router.include_router(documents_router, prefix="/documents", tags=["Documents"])
api_router.include_router(retrieval_router, prefix="/retrieval", tags=["Retrieval"])
api_router.include_router(rag_router, prefix="/rag", tags=["RAG Generation"])
api_router.include_router(conversations_router, prefix="/conversations", tags=["Conversations"])

__all__ = ["api_router"]
