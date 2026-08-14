#!/usr/bin/env python3
"""Create a credential-free Pi handoff ZIP with a SHA-256 sidecar."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = Path.home() / "Downloads"

FILES = [
    "edge_contract_v1.py",
    "edge_outbox_v1.py",
    "edge_node_agent.py",
    "edge_bundle_client.py",
    "edge_bundle_manager.py",
    "edge_temporal_runtime.py",
    "temporal_features.py",
    "temporal_sequence.py",
    "hybrid_fusion.py",
    "motion_watcher.py",
    "person_tracker.py",
    "spatial_geometry.py",
    "config/edge_node_secure.example.json",
    "config/edge_fusion_rpi5_candidate_v1.json",
    "scripts/pull_edge_bundle.py",
    "scripts/probe_rpi_runtime.sh",
    "docs/CENTRAL_PREDEPLOY_STATUS_2026-08-07.md",
    "docs/PI_CANARY_RUNBOOK_V2.md",
    "deploy/dmc-edge-agent.service.example",
]
BUNDLE = "artifacts/edge/bundles/rpi5-onnx-candidate-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = OUTPUT / f"DMC_POSE_edge_handoff_{stamp}.zip"
    selected = [PROJECT / name for name in FILES]
    selected.extend(path for path in sorted((PROJECT / BUNDLE).iterdir()) if path.is_file())
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing handoff files: " + ", ".join(missing))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in selected:
            archive.write(path, Path("DMC_POSE_edge") / path.relative_to(PROJECT))
        archive.writestr("DMC_POSE_edge/SECURITY.txt", (
            "No API token, SSH credential, RTSP credential, runtime frame, or raw video is included.\n"
            "Provision /etc/dmc_pose/edge_api_token through a separate secure channel with mode 0600.\n"
        ))
    digest = sha256(target)
    sidecar = Path(str(target) + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    print(json.dumps({
        "zip": str(target), "sha256_file": str(sidecar), "sha256": digest,
        "bytes": target.stat().st_size, "files": len(selected) + 1,
        "credentials_included": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
