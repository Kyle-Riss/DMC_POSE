#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${DMC_EDGE_PYTHON:-/home/dmc/anaconda3/envs/pose-cuda/bin/python}"
PORT="${DMC_EDGE_PORT:-8020}"

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m uvicorn edge_control_server:app --host 0.0.0.0 --port "$PORT"
