#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${PROJECT_ROOT}/external_datasets/omnifall"
VENV="${PROJECT_ROOT}/external_datasets/.download_venv"

mkdir -p "${DATASET_ROOT}"

if [[ ! -x "${VENV}/bin/python" ]]; then
    python3 -m venv "${VENV}"
fi

if ! "${VENV}/bin/python" -m pip --version >/dev/null 2>&1; then
    "${VENV}/bin/python" -m ensurepip --upgrade
fi

"${VENV}/bin/python" -m pip install --upgrade pip huggingface_hub

"${VENV}/bin/hf" download simplexsigil2/omnifall \
    --repo-type dataset \
    --local-dir "${DATASET_ROOT}" \
    --include "*.parquet" \
    --include "*.csv" \
    --include "*.json" \
    --include "*.md" \
    --include "*.py"

echo "OmniFall metadata download complete."
