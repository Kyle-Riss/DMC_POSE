#!/bin/bash
# Bed 4 (Backup) - 192.168.0.179
export POSE_CAMERA_ID=raspi_bed_004
export POSE_YOLO_DEVICE=0
export POSE_PARALLEL_WORKERS=2
conda activate pose-cuda
echo "🎥 Bed 4 (Backup) 서버 시작 - 192.168.0.179:8554"
python /home/dmc/AI/DMC_POSE/server_parallel.py
