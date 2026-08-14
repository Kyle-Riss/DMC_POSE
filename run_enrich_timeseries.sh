#!/usr/bin/env bash
# Stage B + event detect: v1 timeseries → enriched → left_bed triggers
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/media/dmc/Moredigm1/Dataset/Raw_data}"
if [[ ! -d "$RAW_ROOT/video" ]] && [[ -d /home/dmc/Dataset/Raw_data/video ]]; then
  RAW_ROOT="/home/dmc/Dataset/Raw_data"
fi

IN_DIR="${IN_DIR:-}"
SAMPLE_HZ="${SAMPLE_HZ:-10.0}"
if [[ -z "$IN_DIR" ]]; then
  if [[ "$SAMPLE_HZ" == "1.0" || "$SAMPLE_HZ" == "1" ]]; then
    IN_DIR="$RAW_ROOT/timeseries"
  else
    hz="${SAMPLE_HZ%.*}"
    IN_DIR="$RAW_ROOT/timeseries_${hz}hz"
  fi
fi

if [[ ! -d "$IN_DIR" ]]; then
  echo "입력 폴더 없음: $IN_DIR"
  echo "  먼저: bash run_timeseries_10hz.sh"
  exit 1
fi

source /home/dmc/anaconda3/etc/profile.d/conda.sh
conda activate pose-cuda
cd /home/dmc/AI/DMC_POSE

OUT_DIR="${OUT_DIR:-${IN_DIR}_enriched}"
EXTRA=()
[[ -n "${VIDEO:-}" ]] && EXTRA+=(--video "$VIDEO")
[[ "${NO_VIDEO:-0}" == "1" ]] && EXTRA+=(--no-video)

echo "=== enrich: $IN_DIR → $OUT_DIR ==="
python enrich_timeseries.py \
  --in-dir "$IN_DIR" \
  --out-dir "$OUT_DIR" \
  --raw-root "$RAW_ROOT" \
  --sample-hz "$SAMPLE_HZ" \
  "${EXTRA[@]}"

echo "=== detect events ==="
python detect_timeseries_events.py \
  --in-dir "$OUT_DIR" \
  "${EXTRA[@]}"

echo "Done. See runs/timeseries_events/all_events.json"
