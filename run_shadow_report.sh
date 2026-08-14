#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${POSE_PYTHON_BIN:-/home/dmc/anaconda3/envs/pose-cuda/bin/python}"

cd "$SCRIPT_DIR"

"$PYTHON_BIN" summarize_shadow_features.py \
  --out runtime_data/shadow_summary.json
"$PYTHON_BIN" prepare_shadow_review.py
"$PYTHON_BIN" evaluate_shadow_operations.py "$@"

echo "Phase 9 report: $SCRIPT_DIR/runtime_data/operational_report.json"
