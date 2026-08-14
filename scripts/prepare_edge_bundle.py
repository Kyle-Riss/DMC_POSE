#!/usr/bin/env python3
"""Build a checksum-complete, non-activatable RPi candidate bundle."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BUNDLE_VERSION = "rpi5-onnx-candidate-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    destination = PROJECT / "artifacts/edge/bundles" / BUNDLE_VERSION
    destination.mkdir(parents=True, exist_ok=True)
    sources = [
        ("bed_seg", PROJECT / "bed_seg/runs/bed_seg/weights/best.onnx", "bed_seg.onnx", "onnx"),
        ("pose", PROJECT / "yolo11n-pose.onnx", "yolo11n-pose.onnx", "onnx"),
        ("posture", PROJECT / "artifacts/edge/rpi5/posture_six_v1/posture_six_fp32.tflite", "posture_six_fp32.tflite", "tflite"),
        ("temporal", PROJECT / "artifacts/edge/rpi5/tcn_observed_only_v2/fall_tcn_normalized.onnx", "fall_tcn_normalized.onnx", "onnx"),
        ("fusion_config", PROJECT / "config/edge_fusion_rpi5_candidate_v1.json", "edge_fusion.json", "json"),
    ]
    artifacts = []
    for role, source, filename, model_format in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / filename
        shutil.copy2(source, target)
        artifacts.append({
            "role": role,
            "filename": filename,
            "sha256": sha(target),
            "bytes": target.stat().st_size,
            "format": model_format,
        })
    manifest = {
        "contract_version": "dmc_pose_edge_v1",
        "bundle_version": BUNDLE_VERSION,
        "status": "benchmark_required",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": "rpi5",
        "feature_schema": "pose_temporal_109_v1",
        "sample_hz": 10.0,
        "temporal_rows": 30,
        "artifacts": artifacts,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (PROJECT / "config/edge_model_bundle_candidate_v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
