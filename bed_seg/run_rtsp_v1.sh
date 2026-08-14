#!/usr/bin/env bash
# RTSP 실시간 카메라용 bed-seg v1 파이프라인
set -euo pipefail
source /home/dmc/anaconda3/etc/profile.d/conda.sh
conda activate pose-cuda
cd /home/dmc/AI/DMC_POSE/bed_seg

RTSP_URL="${RTSP_URL:-rtsp://192.168.0.161:8554/stream}"
CAPTURE_COUNT="${CAPTURE_COUNT:-50}"
CAPTURE_INTERVAL="${CAPTURE_INTERVAL:-2.0}"

echo "=== 1) RTSP 프레임 캡처 (${CAPTURE_COUNT}장) ==="
python capture_rtsp_frames.py \
  --url "$RTSP_URL" \
  --count "$CAPTURE_COUNT" \
  --interval-sec "$CAPTURE_INTERVAL"

echo ""
echo "=== 2) 수동 polygon 라벨 (GUI 필요) ==="
echo "  python /home/dmc/labeling/label_bed_polygon.py \\"
echo "    --images $(pwd)/rtsp_raw \\"
echo "    --labels $(pwd)/manual_labels"
echo ""
echo "라벨 완료 후 Enter..."
read -r _

echo "=== 3) dataset_v1 빌드 ==="
python prepare_rtsp_dataset.py

echo "=== 4) v1 학습 ==="
python train_bed_seg.py

echo "=== 5) 완료 ==="
echo "모델: /home/dmc/AI/DMC_POSE/yolo11n-bed-seg.pt"
echo "서버 재시작: bash /home/dmc/AI/DMC_POSE/run_server.sh"
