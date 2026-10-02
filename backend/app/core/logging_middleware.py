"""Structured Logging and Request Correlation Middleware."""

import json
import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("app.access")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming HTTP requests with latency, correlation IDs, and status codes.
    
    Guarantees:
    - Never logs passwords, tokens, API keys, or raw file bodies.
    - Generates or propagates X-Request-ID across all requests.
    - Captures unhandled runtime exceptions with safe sanitized 500 responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id

        start_time = time.perf_counter()
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            response.headers["X-Request-ID"] = request_id

            # Safe access log
            log_data = {
                "event": "http_request",
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "latency_ms": duration_ms,
                "client_ip": client_ip,
            }
            logger.info(json.dumps(log_data))
            return response

        except Exception as e:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            sanitized_error = str(e).replace("\\", "/").split(":")[-1].strip()
            log_data = {
                "event": "http_error",
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": 500,
                "latency_ms": duration_ms,
                "client_ip": client_ip,
                "error": sanitized_error,
            }
            logger.error(json.dumps(log_data))

            return JSONResponse(
                status_code=500,
                content={
                    "detail": "An internal server error occurred.",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
