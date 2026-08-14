#!/bin/bash
# Bed 3 (Testing) - 192.168.0.178
export POSE_CAMERA_ID=raspi_bed_003
export POSE_YOLO_DEVICE=0
export POSE_PARALLEL_WORKERS=2
conda activate pose-cuda
echo "🎥 Bed 3 (Testing) 서버 시작 - 192.168.0.178:8554"
python /home/dmc/AI/DMC_POSE/server_parallel.py
