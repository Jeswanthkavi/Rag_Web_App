"""Health and Readiness probe endpoints."""

import logging
from typing import Any, Dict
from fastapi import APIRouter, Response, status
from backend.app.db.session import check_db_health
from backend.app.vectorstore.client import create_qdrant_client

logger = logging.getLogger("app.health")
router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check() -> Dict[str, str]:
    """Liveness probe to check if the API server process is alive."""
    return {"status": "healthy", "service": "advanced-rag-api"}


@router.get("/ready", tags=["Health"])
def readiness_check(response: Response) -> Dict[str, Any]:
    """Readiness probe to check MySQL database and Qdrant vector store availability."""
    checks = {
        "status": "ready",
        "database": "unknown",
        "vector_store": "unknown",
    }
    is_ready = True

    # 1. Check MySQL Relational Database
    try:
        db_res = check_db_health()
        if db_res.get("status") == "healthy" and db_res.get("is_alive"):
            checks["database"] = "connected"
        else:
            checks["database"] = "unavailable"
            is_ready = False
    except Exception as e:
        logger.warning(f"Readiness check failed for database: {e}")
        checks["database"] = "unavailable"
        is_ready = False

    # 2. Check Qdrant Vector Store
    try:
        qdrant = create_qdrant_client()
        collections_info = qdrant.get_collections()
        if collections_info is not None:
            checks["vector_store"] = "connected"
        else:
            checks["vector_store"] = "unavailable"
            is_ready = False
    except Exception as e:
        logger.warning(f"Readiness check failed for vector store: {e}")
        checks["vector_store"] = "unavailable"
        is_ready = False

    if not is_ready:
        checks["status"] = "not_ready"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return checks
