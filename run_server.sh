#!/usr/bin/env bash
# Pose live server — YOLO GPU + low-latency RTSP
set -euo pipefail

source /home/dmc/anaconda3/etc/profile.d/conda.sh
conda activate pose-cuda

cd /home/dmc/AI/DMC_POSE

export TF_CPP_MIN_LOG_LEVEL=3
export POSE_YOLO_DEVICE="${POSE_YOLO_DEVICE:-0}"      # YOLO seg/pose GPU
export POSE_SEG_EVERY="${POSE_SEG_EVERY:-3}"          # seg every N frames
export POSE_RTSP_URL="${POSE_RTSP_URL:-rtsp://192.168.0.161:8554/stream}"
export POSE_API_HOST="${POSE_API_HOST:-0.0.0.0}"
export POSE_API_PORT="${POSE_API_PORT:-8000}"
export POSE_FRAME_WIDTH="${POSE_FRAME_WIDTH:-640}"   # 640×360 (extracted_frames / RTSP)
export POSE_BED_SEG_CONF="${POSE_BED_SEG_CONF:-0.01}"   # v1 model scores ~0.01–0.02 on live RTSP
export POSE_PRESET="${POSE_PRESET:-approx_seg}"         # dilated zone + ROI fallback
export POSE_USE_BED_ROI="${POSE_USE_BED_ROI:-1}"
# viewer overlay (0=off, 1=on) — 기본은 침대 mask + pose 뼈대만
export POSE_SHOW_BED_ROI="${POSE_SHOW_BED_ROI:-0}"
export POSE_SHOW_BED_ZONES="${POSE_SHOW_BED_ZONES:-0}"
export POSE_SHOW_BED_BBOX="${POSE_SHOW_BED_BBOX:-0}"
export POSE_SHOW_RAIL_ROI="${POSE_SHOW_RAIL_ROI:-0}"
export POSE_YOLO_BOXES="${POSE_YOLO_BOXES:-0}"
export POSE_USE_RAIL="${POSE_USE_RAIL:-1}"
export POSE_RAIL_CONFIG="${POSE_RAIL_CONFIG:-/home/dmc/AI/DMC_POSE/rail/rail_config.json}"

# 기존 서버가 있으면 종료
if ss -tlnp 2>/dev/null | grep -q ":${POSE_API_PORT} "; then
  echo "Stopping existing server on :${POSE_API_PORT}"
  fuser -k "${POSE_API_PORT}/tcp" 2>/dev/null || true
  sleep 1
fi

echo "RTSP: $POSE_RTSP_URL (Main)"
echo "Viewer: http://$(hostname -I | awk '{print $1}'):${POSE_API_PORT}/viewer"
python server_all_cameras.py


