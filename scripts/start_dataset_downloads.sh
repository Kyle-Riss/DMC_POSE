#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${PROJECT_ROOT}/runtime_data/dataset_downloads"

mkdir -p "${RUNTIME_ROOT}"

start_job() {
    local name="$1"
    local script="$2"
    local pid_file="${RUNTIME_ROOT}/${name}.pid"
    local log_file="${RUNTIME_ROOT}/${name}.log"
    local old_pid=""

    if [[ -s "${pid_file}" ]]; then
        old_pid="$(<"${pid_file}")"
        if kill -0 "${old_pid}" 2>/dev/null; then
            echo "[running] ${name}: pid=${old_pid}"
            return 0
        fi
    fi

    nohup bash "${script}" >"${log_file}" 2>&1 &
    echo "$!" >"${pid_file}"
    echo "[started] ${name}: pid=$! log=${log_file}"
}

start_job \
    "fallvision" \
    "${PROJECT_ROOT}/scripts/download_fallvision_files.sh"

start_job \
    "omnifall_metadata" \
    "${PROJECT_ROOT}/scripts/download_omnifall_metadata.sh"
