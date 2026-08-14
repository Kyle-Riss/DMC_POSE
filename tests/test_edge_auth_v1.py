from fastapi import FastAPI
from fastapi.testclient import TestClient

from edge_auth_v1 import EdgeBearerAuthMiddleware


def test_auth_protects_non_health_routes():
    app = FastAPI()
    app.add_middleware(EdgeBearerAuthMiddleware, token="x" * 32)

    @app.get("/health/live")
    def live():
        return {"live": True}

    @app.get("/edge/nodes")
    def nodes():
        return {"nodes": []}

    client = TestClient(app)
    assert client.get("/health/live").status_code == 200
    assert client.get("/edge/nodes").status_code == 401
    assert client.get("/edge/nodes", headers={"Authorization": "Bearer " + "x" * 32}).status_code == 200
