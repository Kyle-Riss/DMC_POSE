#!/usr/bin/env bash
# Raw 8영상 → 10Hz 시계열 (0.1초마다 1행, timeseries_10hz/ 에 저장)
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/media/dmc/Moredigm1/Dataset/Raw_data}"
if [[ ! -d "$RAW_ROOT/video" ]] && [[ -d /home/dmc/Dataset/Raw_data/video ]]; then
  RAW_ROOT="/home/dmc/Dataset/Raw_data"
fi

if [[ ! -d "$RAW_ROOT/video" ]]; then
  echo "Raw_data 없음: $RAW_ROOT"
  echo "  Moredigm USB 마운트 후: RAW_ROOT=/media/dmc/Moredigm1/Dataset/Raw_data bash $0"
  exit 1
fi

source /home/dmc/anaconda3/etc/profile.d/conda.sh
conda activate pose-cuda

export TF_CPP_MIN_LOG_LEVEL=3
DEVICE="${DEVICE:-0}"

cd /home/dmc/AI/DMC_POSE
python extract_raw_timeseries.py \
  --raw-root "$RAW_ROOT" \
  --sample-hz 10.0 \
  --device "$DEVICE" \
  --model /home/dmc/AI/DMC_POSE/my_model_six_check.keras \
  "$@"

echo "완료: $RAW_ROOT/timeseries_10hz/"
