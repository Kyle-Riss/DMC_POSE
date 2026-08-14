#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${PROJECT_ROOT}/runtime_data/dataset_downloads"
PID_FILE="${STATE_DIR}/fallvision_detached.pid"
LOG_FILE="${STATE_DIR}/fallvision_detached.log"

mkdir -p "${STATE_DIR}"

if [[ -s "${PID_FILE}" ]]; then
    existing_pid="$(<"${PID_FILE}")"
    if kill -0 "${existing_pid}" 2>/dev/null; then
        echo "FallVision download already running: pid=${existing_pid}"
        exit 0
    fi
fi

cd "${PROJECT_ROOT}"
nohup setsid bash scripts/download_fallvision_files.sh \
    >>"${LOG_FILE}" 2>&1 </dev/null &
download_pid=$!
echo "${download_pid}" >"${PID_FILE}"

echo "FallVision detached download started"
echo "pid=${download_pid}"
echo "log=${LOG_FILE}"
