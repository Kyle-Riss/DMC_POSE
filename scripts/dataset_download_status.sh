#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${PROJECT_ROOT}/runtime_data/dataset_downloads"

for name in fallvision omnifall_metadata; do
    pid_file="${RUNTIME_ROOT}/${name}.pid"
    log_file="${RUNTIME_ROOT}/${name}.log"
    pid=""
    state="NOT_STARTED"

    if [[ -s "${pid_file}" ]]; then
        pid="$(<"${pid_file}")"
        if kill -0 "${pid}" 2>/dev/null; then
            state="RUNNING"
        else
            state="STOPPED"
        fi
    fi

    echo "${name}: ${state}${pid:+ pid=${pid}}"
    if [[ -f "${log_file}" ]]; then
        tail -n 8 "${log_file}"
    fi
    echo
done

du -sh \
    "${PROJECT_ROOT}/external_datasets/fallvision" \
    "${PROJECT_ROOT}/external_datasets/omnifall" \
    2>/dev/null || true
