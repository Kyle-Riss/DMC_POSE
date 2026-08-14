#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${PROJECT_ROOT}/external_datasets/fallvision"
METADATA="${DATASET_ROOT}/metadata.json"
OUTPUT_ROOT="${DATASET_ROOT}/files"
PARALLEL="${FALLVISION_PARALLEL:-3}"

if [[ ! -s "${METADATA}" ]]; then
    echo "missing metadata: ${METADATA}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

download_one() {
    local file_id="$1"
    local label="$2"
    local expected_size="$3"
    local destination="${OUTPUT_ROOT}/${label}"
    local partial="${destination}.part"
    local actual_size

    mkdir -p "$(dirname "${destination}")"

    if [[ -f "${destination}" ]]; then
        actual_size="$(stat -c '%s' "${destination}")"
        if [[ "${actual_size}" == "${expected_size}" ]]; then
            echo "[skip] ${label} (${actual_size} bytes)"
            return 0
        fi
        echo "[invalid existing] ${label}: ${actual_size}/${expected_size}" >&2
        return 1
    fi

    echo "[download] ${label} (${expected_size} bytes)"
    curl -fL \
        --retry 10 \
        --retry-delay 5 \
        --retry-all-errors \
        --continue-at - \
        "https://dataverse.harvard.edu/api/access/datafile/${file_id}?format=original" \
        --output "${partial}"

    actual_size="$(stat -c '%s' "${partial}")"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
        echo "[size mismatch] ${label}: ${actual_size}/${expected_size}" >&2
        return 1
    fi

    mv "${partial}" "${destination}"
    echo "[complete] ${label}"
}

declare -a batch_pids=()
failed=0

wait_batch() {
    local pid
    for pid in "${batch_pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    batch_pids=()
}

while IFS=$'\t' read -r file_id label expected_size; do
    download_one "${file_id}" "${label}" "${expected_size}" &
    batch_pids+=("$!")
    if (( ${#batch_pids[@]} >= PARALLEL )); then
        wait_batch
    fi
done < <(
    jq -r '
        .data.latestVersion.files[]
        | [.dataFile.id, .label, .dataFile.filesize]
        | @tsv
    ' "${METADATA}"
)

if (( ${#batch_pids[@]} > 0 )); then
    wait_batch
fi

if (( failed != 0 )); then
    echo "FallVision completed with one or more failed files." >&2
    exit 1
fi

echo "FallVision individual-file download complete."
