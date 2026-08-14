#!/usr/bin/env python3
"""Launch the authenticated edge control plane without exposing its token."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--token-file", type=Path, default=Path("runtime_data/edge_control/api_token"))
    args = parser.parse_args()
    token = args.token_file.resolve().read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise RuntimeError("edge API token is missing or too short")
    os.environ["DMC_EDGE_API_TOKEN"] = token
    uvicorn.run("edge_control_server_secure:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
