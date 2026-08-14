#!/bin/bash
# 병렬 GPU 추론 서버 실행
# RTSP + 병렬 seg/pose + 6-class 분류

export POSE_YOLO_DEVICE=0
export POSE_SEG_EVERY=3
export POSE_PARALLEL_WORKERS=2
export POSE_FRAME_QUEUE_SIZE=30
export POSE_RTSP_URL="rtsp://192.168.0.161:8554/stream"
export POSE_FRAME_WIDTH=640
export POSE_USE_BED_ROI=1

conda activate pose-cuda

echo "🚀 Parallel GPU Pipeline Server 시작..."
echo "   YOLO Device: $POSE_YOLO_DEVICE"
echo "   Parallel Workers: $POSE_PARALLEL_WORKERS"
echo "   RTSP: $POSE_RTSP_URL"
echo "   Viewer: http://$(hostname -I | awk '{print $1}'):8000/viewer"

python /home/dmc/AI/DMC_POSE/server_parallel.py
