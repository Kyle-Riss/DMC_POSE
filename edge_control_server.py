"""Independent control-plane API for bed-side DMC_POSE edge nodes."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from edge_contract_v1 import (
    EDGE_CONTRACT_VERSION,
    EdgeEventEnd,
    EdgeEventStart,
    EdgeHeartbeat,
    EdgeInferenceResult,
    EdgeModelBundle,
    utc_now,
)
from edge_registry_v1 import EdgeRegistry, EventLifecycleError, SequenceRegression

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "edge_model_bundle_v1.json"
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "runtime_data" / "edge_control"


def load_manifest(path: str | Path) -> EdgeModelBundle:
    with Path(path).open(encoding="utf-8") as handle:
        return EdgeModelBundle.model_validate(json.load(handle))


def create_app(
    *,
    runtime_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> FastAPI:
    data_dir = Path(runtime_dir or os.environ.get("DMC_EDGE_RUNTIME_DIR", DEFAULT_RUNTIME_DIR))
    bundle_path = Path(manifest_path or os.environ.get("DMC_EDGE_MANIFEST", DEFAULT_MANIFEST))
    registry = EdgeRegistry(data_dir)
    manifest = load_manifest(bundle_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry.start()
        yield
        registry.stop()

    app = FastAPI(
        title="DMC_POSE Edge Control Plane",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.registry = registry
    app.state.manifest = manifest

    def response(result) -> dict:
        return {
            "accepted": result.accepted,
            "duplicate": result.duplicate,
            "server_time": utc_now().isoformat(),
            "desired_bundle_version": manifest.bundle_version,
            "bundle_status": manifest.status,
        }

    @app.get("/health/live")
    def health_live() -> dict:
        return {"live": True, "contract_version": EDGE_CONTRACT_VERSION}

    @app.get("/health/ready")
    def health_ready(request: Request) -> dict:
        state = request.app.state.registry.snapshot()
        return {
            "ready": state["writer"]["alive"],
            "writer": state["writer"],
            "bundle_status": manifest.status,
        }

    @app.post("/edge/heartbeat")
    def heartbeat(payload: EdgeHeartbeat, request: Request) -> dict:
        try:
            result = request.app.state.registry.accept_heartbeat(payload)
        except SequenceRegression as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return response(result)

    @app.post("/edge/results")
    def inference_result(payload: EdgeInferenceResult, request: Request) -> dict:
        try:
            result = request.app.state.registry.accept_result(payload)
        except SequenceRegression as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return response(result)

    @app.post("/events/start")
    def event_start(payload: EdgeEventStart, request: Request) -> dict:
        try:
            result = request.app.state.registry.start_event(payload)
        except EventLifecycleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        body = response(result)
        body["upload_event_frames"] = payload.event_type in {"FALL", "BED_EXIT_FALL"}
        return body

    @app.post("/events/end")
    def event_end(payload: EdgeEventEnd, request: Request) -> dict:
        try:
            result = request.app.state.registry.end_event(payload)
        except EventLifecycleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return response(result)

    @app.get("/edge/nodes")
    def nodes(request: Request) -> dict:
        return request.app.state.registry.snapshot()

    @app.get("/edge/model-manifest", response_model=EdgeModelBundle)
    def model_manifest(request: Request) -> EdgeModelBundle:
        return request.app.state.manifest

    return app


app = create_app()

