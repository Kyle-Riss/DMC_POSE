#!/usr/bin/env bash
# Phase M0 — capture merge baseline (pose-sixclass vs fall_monitor)
set -euo pipefail

source /home/dmc/anaconda3/etc/profile.d/conda.sh
conda activate pose-cuda

cd /home/dmc/AI/DMC_POSE

export TF_CPP_MIN_LOG_LEVEL=3
export POSE_PRESET="${POSE_PRESET:-approx_seg}"
export POSE_YOLO_DEVICE="${POSE_YOLO_DEVICE:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-runs/merge_baseline/${STAMP}}"
RTSP="${POSE_RTSP_URL:-rtsp://192.168.0.161:8554/stream}"
MODE="${MODE:-images}"   # images | rtsp | status
DURATION="${DURATION:-120}"
MAX_IMAGES="${MAX_IMAGES:-30}"

mkdir -p "$(dirname "$OUT")"

echo "=== Phase M0 baseline ==="
echo "MODE=$MODE  OUT=$OUT  PRESET=$POSE_PRESET"

case "$MODE" in
  images)
    python scripts/merge_baseline_capture.py \
      --image-dir bed_seg/rtsp_raw \
      --max-images "$MAX_IMAGES" \
      --out "$OUT"
    ;;
  rtsp)
    python scripts/merge_baseline_capture.py \
      --rtsp "$RTSP" \
      --duration "$DURATION" \
      --sample-hz 2 \
      --out "$OUT"
    ;;
  status)
    # Terminal 1: bash run_server.sh
    python scripts/merge_baseline_capture.py \
      --status-url "http://127.0.0.1:${POSE_API_PORT:-8000}/status" \
      --duration "$DURATION" \
      --sample-hz 2 \
      --out "$OUT"
    ;;
  full)
    python scripts/merge_baseline_capture.py \
      --image-dir bed_seg/rtsp_raw \
      --max-images "$MAX_IMAGES" \
      --out "${OUT}/offline"
    if python scripts/merge_baseline_capture.py --rtsp "$RTSP" --duration 30 --out "${OUT}/rtsp" 2>/dev/null; then
      echo "RTSP capture OK"
    else
      echo "RTSP capture skipped (camera offline)"
    fi
    ;;
  *)
    echo "Unknown MODE=$MODE (images|rtsp|status|full)"
    exit 1
    ;;
esac

echo "Done: $OUT"
echo "  summary: $OUT/baseline_summary.json"
echo "  frames:  $OUT/baseline_*.jsonl"
