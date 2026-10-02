"""FastAPI Application entry point for Advanced RAG Platform."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
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
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
    )

    # 1. Security Headers Middleware (OWASP protection)
    if settings.security.enable_security_headers:
        from backend.app.core.security_middleware import SecurityHeadersMiddleware
        app.add_middleware(SecurityHeadersMiddleware)

    # 2. Content Length / Payload Bound Middleware (DoS protection)
    from backend.app.core.security_middleware import ContentLengthLimitMiddleware
    app.add_middleware(
        ContentLengthLimitMiddleware,
        max_bytes=settings.security.max_request_body_bytes,
    )

    # 3. Sliding-window Rate Limiting Middleware (Brute-force and flood protection)
    if settings.security.rate_limit_enabled:
        from backend.app.core.security_middleware import RateLimitMiddleware
        app.add_middleware(
            RateLimitMiddleware,
            login_limit=settings.security.rate_limit_login,
            register_limit=settings.security.rate_limit_register,
            upload_limit=settings.security.rate_limit_upload,
            rag_limit=settings.security.rate_limit_rag,
            general_limit=settings.security.rate_limit_general,
        )

    # 4. Structured Logging & Request Correlation Middleware
    from backend.app.core.logging_middleware import StructuredLoggingMiddleware
    app.add_middleware(StructuredLoggingMiddleware)

    # Mount API version 1 router
    app.include_router(api_router, prefix=settings.app.api_v1_prefix)

    @app.get("/health", tags=["Health"])
    def root_health():
        return {"status": "healthy", "service": settings.app.app_name}

    @app.get("/ready", tags=["Health"])
    def root_ready(response: Response):
        from backend.app.api.v1.health import readiness_check
        return readiness_check(response=response)

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
