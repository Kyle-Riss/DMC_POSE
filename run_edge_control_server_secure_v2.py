#!/usr/bin/env python3
"""Launch secure v2 with local secret and checksum-verified candidate bundle."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

PROJECT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    token_file = PROJECT / "runtime_data/edge_control/api_token"
    token = token_file.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise RuntimeError("edge API token is missing or too short")
    os.environ["DMC_EDGE_API_TOKEN"] = token
    os.environ["DMC_EDGE_MANIFEST"] = str(PROJECT / "config/edge_model_bundle_candidate_v1.json")
    os.environ["DMC_EDGE_BUNDLE_DIR"] = str(PROJECT / "artifacts/edge/bundles/rpi5-onnx-candidate-v1")
    uvicorn.run("edge_control_server_secure_v2:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
