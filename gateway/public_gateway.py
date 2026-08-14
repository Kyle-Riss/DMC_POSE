"""Minimal public boundary for the protected monitoring core.

The private core remains bound to loopback.  This process is the only HTTP
surface exposed to the LAN and deliberately returns no model, policy, score,
feature, or intermediate-state fields.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse


CORE_URL = os.environ.get("DMC_PRIVATE_CORE_URL", "http://127.0.0.1:18000").rstrip("/")
CAMERA_IDS = tuple(
    item.strip()
    for item in os.environ.get(
        "DMC_PUBLIC_CAMERA_IDS",
        "bed_161,bed_162,bed_174,bed_175,bed_178,bed_179",
    ).split(",")
    if item.strip()
)
TIMEOUT_SEC = float(os.environ.get("DMC_GATEWAY_TIMEOUT_SEC", "3"))
CHUNK_SIZE = int(os.environ.get("DMC_GATEWAY_STREAM_CHUNK", "65536"))

app = FastAPI(
    title="DMC Monitoring API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _request(path: str, *, timeout: float | None = None):
    request = Request(
        f"{CORE_URL}{path}",
        headers={"Accept": "application/json,image/jpeg,multipart/x-mixed-replace"},
    )
    try:
        return urlopen(request, timeout=timeout or TIMEOUT_SEC)
    except HTTPError as exc:
        if exc.code == 404:
            raise HTTPException(status_code=404, detail="not found") from None
        raise HTTPException(status_code=503, detail="service unavailable") from None
    except (URLError, TimeoutError, OSError):
        raise HTTPException(status_code=503, detail="service unavailable") from None


def _json(path: str) -> Any:
    with _request(path) as response:
        try:
            return json.loads(response.read())
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPException(status_code=503, detail="service unavailable") from None


def _contains_alert(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).upper()
    return any(token in text for token in ("FALL", "CRITICAL", "DANGER", "ALERT"))


def _public_state(camera: dict[str, Any]) -> str:
    alert_fields = (
        "fall_status",
        "fall_level",
        "risk_level",
        "fusion_phase",
        "bed_event",
    )
    if any(_contains_alert(camera.get(field)) for field in alert_fields):
        return "ALERT"

    person_count = camera.get("person_count", 0)
    try:
        if int(person_count or 0) > 0:
            return "MONITORING"
    except (TypeError, ValueError):
        pass
    return "EMPTY"


def _updated_at(camera: dict[str, Any]) -> str:
    for field in ("updated_at", "timestamp", "last_update", "frame_timestamp"):
        value = camera.get(field)
        if value is not None:
            return str(value)
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@app.get("/health/live")
def health_live() -> dict[str, bool]:
    return {"live": True}


@app.get("/health/ready")
def health_ready() -> dict[str, bool]:
    try:
        raw = _json("/health/ready")
        return {"ready": bool(raw.get("ready")) if isinstance(raw, dict) else False}
    except HTTPException:
        return {"ready": False}


@app.get("/status")
def status() -> dict[str, dict[str, str]]:
    raw = _json("/status")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=503, detail="service unavailable")

    public: dict[str, dict[str, str]] = {}
    for camera_id in CAMERA_IDS:
        camera = raw.get(camera_id)
        if not isinstance(camera, dict):
            camera = {}
        public[camera_id] = {
            "camera_id": camera_id,
            "state": _public_state(camera),
            "updated_at": _updated_at(camera),
        }
    return public


@app.get("/image/{camera_id}")
def image(camera_id: str) -> Response:
    if camera_id not in CAMERA_IDS:
        raise HTTPException(status_code=404, detail="not found")
    with _request(f"/image/{camera_id}") as upstream:
        payload = upstream.read()
        content_type = upstream.headers.get("content-type", "image/jpeg")
    return Response(content=payload, media_type=content_type)


def _stream(camera_id: str) -> Iterator[bytes]:
    upstream = _request(f"/video/{camera_id}", timeout=30.0)
    try:
        while True:
            chunk = upstream.read(CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
    finally:
        upstream.close()


@app.get("/video/{camera_id}")
def video(camera_id: str) -> StreamingResponse:
    if camera_id not in CAMERA_IDS:
        raise HTTPException(status_code=404, detail="not found")
    return StreamingResponse(
        _stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/viewer", response_class=HTMLResponse)
def viewer() -> str:
    cards = "".join(
        f'<article><header><span>{camera_id}</span><b id="s-{camera_id}">--</b></header>'
        f'<img src="/video/{camera_id}" alt="{camera_id}"></article>'
        for camera_id in CAMERA_IDS
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DMC Monitoring</title>
<style>
body{{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;padding:12px}}
article{{background:#1d1d1d;border-radius:8px;overflow:hidden}}
header{{display:flex;justify-content:space-between;padding:10px 12px}}
img{{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:#000}}
b{{color:#8fd3ff}} b.alert{{color:#ff6565}}
</style></head><body><main>{cards}</main>
<script>
async function refresh(){{
  try{{
    const data=await fetch('/status',{{cache:'no-store'}}).then(r=>r.json());
    for(const [id,row] of Object.entries(data)){{
      const el=document.getElementById('s-'+id); if(!el) continue;
      el.textContent=row.state; el.className=row.state==='ALERT'?'alert':'';
    }}
  }}catch(_e){{}}
}}
refresh(); setInterval(refresh,1000);
</script></body></html>"""
