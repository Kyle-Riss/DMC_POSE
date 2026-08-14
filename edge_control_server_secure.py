"""Authenticated edge control-plane application."""
from __future__ import annotations

import os

from edge_auth_v1 import EdgeBearerAuthMiddleware
from edge_control_server import create_app


def create_secure_app():
    app = create_app()
    app.add_middleware(EdgeBearerAuthMiddleware, token=os.environ.get("DMC_EDGE_API_TOKEN"))
    app.title = "DMC_POSE Secure Edge Control Plane"
    app.version = "1.1.0"
    return app


app = create_secure_app()
