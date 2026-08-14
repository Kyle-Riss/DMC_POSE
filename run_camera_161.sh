#!/bin/bash
# Bed 1 (Main) - 192.168.0.161
export POSE_CAMERA_ID=raspi_bed_001
export POSE_YOLO_DEVICE=0
export POSE_PARALLEL_WORKERS=2
conda activate pose-cuda
echo "🎥 Bed 1 (Main) 서버 시작 - 192.168.0.161:8554"
python /home/dmc/AI/DMC_POSE/server_parallel.py
