"""Authenticated control plane with verified artifact and evidence APIs."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from edge_auth_v1 import EdgeBearerAuthMiddleware
from edge_bundle_manager import BundleVerificationError, verify_bundle_files
from edge_control_server import create_app
from edge_event_frame_store import EventFrameStore, EvidenceFrameError
from edge_registry_v1 import EventLifecycleError

PROJECT = Path(__file__).resolve().parent


def create_secure_app_v2():
    app = create_app()
    token = os.environ.get("DMC_EDGE_API_TOKEN")
    app.add_middleware(EdgeBearerAuthMiddleware, token=token)
    bundle_dir = Path(os.environ.get(
        "DMC_EDGE_BUNDLE_DIR",
        PROJECT / "artifacts/edge/bundles/rpi5-onnx-candidate-v1",
    )).resolve()
    try:
        verify_bundle_files(app.state.manifest, bundle_dir)
    except BundleVerificationError as exc:
        raise RuntimeError(f"edge bundle verification failed: {exc}") from exc
    frame_store = EventFrameStore(
        Path(os.environ.get("DMC_EDGE_EVENT_FRAMES", PROJECT / "runtime_data/edge_event_frames"))
    )
    app.state.bundle_dir = bundle_dir
    app.state.frame_store = frame_store
    app.title = "DMC_POSE Secure Edge Control Plane"
    app.version = "1.2.0"

    @app.get("/edge/artifacts/{filename}")
    def artifact(filename: str, request: Request):
        allowed = {item.filename: item for item in request.app.state.manifest.artifacts}
        spec = allowed.get(filename)
        if spec is None:
            raise HTTPException(status_code=404, detail="artifact is not in the active manifest")
        path = (request.app.state.bundle_dir / filename).resolve()
        if path.parent != request.app.state.bundle_dir or not path.is_file():
            raise HTTPException(status_code=404, detail="artifact unavailable")
        return FileResponse(path, filename=filename, headers={"X-Artifact-SHA256": spec.sha256})

    @app.put("/events/{event_id}/frames/{frame_seq}")
    async def event_frame(event_id: str, frame_seq: int, request: Request, node_id: str, camera_id: str):
        try:
            request.app.state.registry.validate_open_event_owner(event_id, node_id, camera_id)
        except EventLifecycleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "image/jpeg":
            raise HTTPException(status_code=415, detail="image/jpeg required")
        declared = request.headers.get("content-length")
        if declared and int(declared) > request.app.state.frame_store.max_frame_bytes:
            raise HTTPException(status_code=413, detail="frame too large")
        body = await request.body()
        try:
            stored = request.app.state.frame_store.put(
                event_id=event_id, node_id=node_id, camera_id=camera_id,
                frame_seq=frame_seq, jpeg=body,
            )
        except EvidenceFrameError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "accepted": True, "duplicate": stored.duplicate,
            "sha256": stored.sha256, "bytes": stored.bytes,
        }

    return app


app = create_secure_app_v2()
