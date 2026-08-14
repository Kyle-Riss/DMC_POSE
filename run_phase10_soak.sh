#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DURATION_SEC="${POSE_SOAK_DURATION_SEC:-3600}"
INTERVAL_SEC="${POSE_SOAK_INTERVAL_SEC:-5}"

exec /home/dmc/anaconda3/envs/pose-cuda/bin/python phase10_soak.py \
  --duration-sec "$DURATION_SEC" \
  --interval-sec "$INTERVAL_SEC"
