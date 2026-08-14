#!/usr/bin/env python3
"""Non-destructive end-to-end smoke test for the secure edge control plane."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_bundle_client import EdgeBundleClient


def status(request):
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8020")
    parser.add_argument("--token-file", default="runtime_data/edge_control/api_token")
    args = parser.parse_args()
    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    auth = {"Authorization": f"Bearer {token}"}

    unauthorized, _ = status(urllib.request.Request(args.server + "/edge/model-manifest"))
    authorized, payload = status(urllib.request.Request(
        args.server + "/edge/model-manifest", headers=auth))
    manifest = json.loads(payload) if authorized == 200 else {}

    missing, _ = status(urllib.request.Request(
        args.server + "/edge/artifacts/not-in-manifest.bin", headers=auth))
    event_id = "smoke-" + uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    start_payload = {
        "event_id": event_id, "node_id": "smoke-node", "camera_id": "bed_161",
        "boot_id": "smoke-boot", "started_at": now, "start_frame_seq": 1,
        "event_type": "CANDIDATE", "model_bundle_version": manifest.get("bundle_version"),
        "roi_version": 1, "pre_event_frames_available": 1,
        "pre_event_coverage_sec": 0.1, "peak_risk": 0.7, "evidence": ["smoke"],
    }
    start_request = urllib.request.Request(
        args.server + "/events/start",
        data=json.dumps(start_payload).encode(), headers={**auth, "Content-Type": "application/json"},
        method="POST")
    event_start, _ = status(start_request)
    query = urllib.parse.urlencode({"node_id": "smoke-node", "camera_id": "bed_161"})
    invalid_request = urllib.request.Request(
        f"{args.server}/events/{event_id}/frames/1?{query}", data=b"not-jpeg",
        headers={**auth, "Content-Type": "image/jpeg"}, method="PUT")
    invalid_jpeg, _ = status(invalid_request)
    valid_request = urllib.request.Request(
        f"{args.server}/events/{event_id}/frames/2?{query}", data=b"\xff\xd8\xff\xd9",
        headers={**auth, "Content-Type": "image/jpeg"}, method="PUT")
    valid_jpeg, valid_body = status(valid_request)
    valid_response = json.loads(valid_body) if valid_jpeg == 200 else {}
    wrong_owner_query = urllib.parse.urlencode({"node_id": "wrong-node", "camera_id": "bed_161"})
    wrong_owner_request = urllib.request.Request(
        f"{args.server}/events/{event_id}/frames/3?{wrong_owner_query}", data=b"\xff\xd8\xff\xd9",
        headers={**auth, "Content-Type": "image/jpeg"}, method="PUT")
    wrong_owner, _ = status(wrong_owner_request)
    end_payload = {
        "event_id": event_id, "node_id": "smoke-node", "camera_id": "bed_161",
        "boot_id": "smoke-boot", "ended_at": datetime.now(timezone.utc).isoformat(),
        "end_frame_seq": 2, "peak_risk": 0.7, "uploaded_frame_count": 1,
        "close_reason": "completed",
    }
    end_request = urllib.request.Request(
        args.server + "/events/end", data=json.dumps(end_payload).encode(),
        headers={**auth, "Content-Type": "application/json"}, method="POST")
    event_end, _ = status(end_request)

    with tempfile.TemporaryDirectory(prefix="dmc-edge-smoke-") as root:
        client = EdgeBundleClient(args.server, token, root)
        bundle, installed = client.download_and_install(activate=False)
        staged_files = sorted(path.name for path in installed.iterdir())

    result = {
        "unauthorized_manifest_status": unauthorized,
        "authorized_manifest_status": authorized,
        "unknown_artifact_status": missing,
        "invalid_jpeg_status": invalid_jpeg,
        "event_start_status": event_start,
        "valid_jpeg_status": valid_jpeg,
        "valid_jpeg_duplicate": valid_response.get("duplicate"),
        "wrong_owner_status": wrong_owner,
        "event_end_status": event_end,
        "bundle_version": manifest.get("bundle_version"),
        "bundle_status": manifest.get("status"),
        "staged_artifact_count": len(staged_files) - int("manifest.json" in staged_files),
        "activation_attempted": False,
        "ok": unauthorized == 401 and authorized == 200 and missing == 404
              and event_start == 200 and invalid_jpeg == 409 and valid_jpeg == 200
              and wrong_owner == 409 and event_end == 200
              and bundle.bundle_version == manifest.get("bundle_version"),
    }
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
