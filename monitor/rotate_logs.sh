#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(dirname "$0")"
cd "$BASE_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
for f in log.csv alerts.log; do
  if [ -f "$f" ]; then
    mv "$f" "${f}.${TS}"
    gzip -f "${f}.${TS}" || true
  fi
done
# keep only 7 most recent files
ls -1t *.log.*.gz 2>/dev/null | tail -n +8 | xargs -r rm -f
ls -1t *.csv.*.gz 2>/dev/null | tail -n +8 | xargs -r rm -f
# ensure fresh files exist
: > log.csv
: > alerts.log
