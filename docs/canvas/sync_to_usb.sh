#!/usr/bin/env bash
# Canvas 파일 → USB Dataset/docs/canvas/
set -euo pipefail

USB_MOUNT="${USB_MOUNT:-/media/dmc/Moredigm1}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
CANVAS_SRC="/home/dmc/.cursor/projects/home-dmc/canvases"
DEST="$USB_MOUNT/Dataset/docs/canvas"

if ! mountpoint -q "$USB_MOUNT"; then
  echo "USB 미마운트: $USB_MOUNT"
  echo "  sudo mount -t ntfs3 -o uid=1000,gid=1000 /dev/sda1 $USB_MOUNT"
  exit 1
fi

mkdir -p "$DEST"
cp -v "$SRC_DIR"/*.canvas.tsx "$DEST/" 2>/dev/null || true
for f in pose-pipeline-flow pose-raw-timeseries-data pose-e2e-validation; do
  if [[ -f "$CANVAS_SRC/${f}.canvas.tsx" ]]; then
    cp -v "$CANVAS_SRC/${f}.canvas.tsx" "$DEST/"
  fi
done
cp -v "$SRC_DIR/README.md" "$DEST/"
echo "완료: $DEST"
ls -la "$DEST"
