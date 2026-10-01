"""FastAPI Application entry point for Advanced RAG Platform."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import get_settings, validate_startup_config
from backend.app.api.v1 import api_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup validation and graceful shutdown."""
    # Validate required configuration on startup
    validate_startup_config(settings)
    yield


def create_application() -> FastAPI:
    """Application factory for FastAPI instance."""
    app = FastAPI(
        title=settings.app.app_name,
        description="Production-grade Multi-Tenant Advanced RAG Application Backend",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API version 1 router
    app.include_router(api_router, prefix=settings.app.api_v1_prefix)

    @app.get("/", tags=["Root"])
    def root():
        return {
            "name": settings.app.app_name,
            "status": "online",
            "version": "1.0.0",
            "docs": "/docs",
        }

    return app


app = create_application()
