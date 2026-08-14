"""Bearer-token middleware for the edge control plane."""
from __future__ import annotations

import hmac
import os

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class EdgeBearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, token: str | None = None):
        super().__init__(app)
        self.token = token or os.environ.get("DMC_EDGE_API_TOKEN", "")
        if len(self.token) < 32:
            raise RuntimeError("DMC_EDGE_API_TOKEN must contain at least 32 characters")

    async def dispatch(self, request: Request, call_next):
        if request.url.path in {"/health/live", "/health/ready"}:
            return await call_next(request)
        authorization = request.headers.get("authorization", "")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        if not supplied or not hmac.compare_digest(supplied, self.token):
            return JSONResponse(
                status_code=401,
                content={"detail": "valid edge bearer token required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)
