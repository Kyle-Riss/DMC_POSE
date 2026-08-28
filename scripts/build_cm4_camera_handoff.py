#!/usr/bin/env python3
"""Build a credential-free CM4 camera-appliance handoff archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
FILES = (
    "edge_contract_v1.py",
    "edge_outbox_v1.py",
    "edge_motion_watcher.py",
    "edge_site_runtime.py",
    "edge_node_agent.py",
    "config/edge_node_bed_161_camera_appliance.json",
    "config/edge_roi_profile.example.json",
    "scripts/calibrate_edge_scene.py",
    "scripts/probe_rpi_runtime.sh",
    "deploy/systemd/dmc-pose-camera-appliance.service",
    "docs/CM4_CAMERA_APPLIANCE_V1.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT / "artifacts/edge/handoff/cm4-camera-appliance-v1",
    )
    args = parser.parse_args()
    selected = [PROJECT / item for item in FILES]
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing handoff files: " + ", ".join(missing))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = args.out_dir / f"dmc_cm4_camera_appliance_v1_{stamp}.zip"
    manifest = {
        "schema_version": "dmc_cm4_camera_handoff_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authority": "camera_telemetry_only",
        "final_fall_authority": False,
        "permanent_local_decode_consumers_added": 1,
        "ring_monitor_rtsp_consumers_added": 0,
        "credentials_included": False,
        "files": [],
    }
    for path in selected:
        manifest["files"].append({
            "path": str(path.relative_to(PROJECT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in selected:
            archive.write(path, Path("dmc_cm4_camera_appliance") / path.relative_to(PROJECT))
        archive.writestr(
            "dmc_cm4_camera_appliance/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        archive.writestr(
            "dmc_cm4_camera_appliance/SECURITY.txt",
            "No API token, SSH credential, RTSP credential, frame, or video is included.\n"
            "Provision the API token separately with mode 0600.\n",
        )
    digest = sha256(target)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    report = {
        **manifest,
        "zip": str(target),
        "zip_bytes": target.stat().st_size,
        "zip_sha256": digest,
        "sha256_file": str(sidecar),
    }
    report_path = target.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
