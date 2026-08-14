#!/usr/bin/env bash
# RTSP health check from dev PC + recovery hints for 192.168.0.161
set -euo pipefail

HOST="${POSE_RTSP_HOST:-192.168.0.161}"
PORT="${POSE_RTSP_PORT:-8554}"
PATH_NAME="${POSE_RTSP_PATH:-stream}"
URL="rtsp://${HOST}:${PORT}/${PATH_NAME}"

echo "=== RTSP diagnose ==="
echo "target: $URL"
echo "this host: $(hostname -I 2>/dev/null | awk '{print $1}')"
echo

echo "--- local port conflict (dev PC should be empty on 8554) ---"
if ss -tlnp 2>/dev/null | grep -E ':8554|:8000' ; then
  echo "WARN: something listens on 8554/8000 on THIS machine"
else
  echo "OK: no local 8554/8000 listener"
fi
ps aux | grep -E 'mediamtx|ffmpeg.*rtsp|run_server' | grep -v grep || echo "OK: no local mediamtx/ffmpeg/rtsp publisher"
echo

echo "--- reachability .161 ---"
ping -c 2 -W 1 "$HOST" | tail -2 || true
for p in 22 80 1935 8554 9997; do
  timeout 1 bash -c "echo > /dev/tcp/${HOST}/${p}" 2>/dev/null && echo "port $p: OPEN" || echo "port $p: closed"
done
echo

echo "--- RTSP server banner ---"
python3 - <<PY
import socket
s = socket.socket(); s.settimeout(3); s.connect(("${HOST}", ${PORT}))
s.sendall(b"OPTIONS rtsp://${HOST}:${PORT}/${PATH_NAME} RTSP/1.0\r\nCSeq: 1\r\n\r\n")
print(s.recv(4096).decode("utf-8", errors="replace"))
s.close()
PY
echo

echo "--- ffprobe /stream ---"
if command -v ffprobe >/dev/null; then
  ffprobe -v error -rtsp_transport tcp \
    -show_entries stream=codec_name,width,height,r_frame_rate \
    -of default=noprint_wrappers=1 "$URL" 2>&1 || true
else
  echo "ffprobe not in PATH (conda activate pose-cuda)"
fi
echo

cat <<'EOF'
=== interpretation ===
- 8554 OPEN + gortsplib + DESCRIBE 404  → relay (MediaMTX) alive, NO publisher on /stream
- 8554 closed / connection refused      → relay down or wrong host
- 400 on wrong path                     → /stream path is correct (configured)
- dev PC 8554 listener                  → unlikely conflict; pose server uses :8000 only

=== fix on 192.168.0.161 (SSH required) ===
ssh dmc@192.168.0.161

# 1) port / process overlap
sudo ss -tlnp | grep -E '8554|1935|554'
ps aux | grep -E 'mediamtx|ffmpeg|gortsplib|v4l2' | grep -v grep
systemctl list-units --type=service | grep -iE 'media|rtsp|stream|ffmpeg'

# 2) typical stack: MediaMTX + ffmpeg publisher
#    restart relay then publisher (adjust unit names if different)
sudo systemctl restart mediamtx 2>/dev/null || sudo systemctl restart rtsp-simple-server 2>/dev/null || true
sudo systemctl restart rtsp-publisher ffmpeg-stream camera-stream 2>/dev/null || true

# 3) if manual ffmpeg (USB cam → RTMP → /stream)
# ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency \
#   -f flv rtmp://127.0.0.1:1935/stream

# 4) verify on .161
ffprobe -rtsp_transport tcp -show_entries stream=width,height \
  -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/stream

=== after RTSP OK on dev PC ===
cd /home/dmc/AI/DMC_POSE
POSE_PRESET=approx_seg bash run_server.sh
ffprobe -rtsp_transport tcp -of default=noprint_wrappers=1 rtsp://192.168.0.161:8554/stream
EOF
