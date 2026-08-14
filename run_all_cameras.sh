#!/bin/bash
# 모든 카메라 병렬 추론 서버
# 6개 카메라 (161, 162, 174, 175, 178, 179) 동시 분석

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Hold this descriptor for the entire server lifetime.  Uvicorn can stop
# accepting HTTP connections before camera-analysis threads have fully exited;
# without a process lock a second invocation can then duplicate all six camera
# pipelines while the old process is still consuming GPU/CPU.
mkdir -p "$SCRIPT_DIR/runtime_data"
SERVER_LOCK_FILE="$SCRIPT_DIR/runtime_data/server_all_cameras.lock"
exec 9>"$SERVER_LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another DMC_POSE camera server is still running or shutting down." >&2
    echo "Lock: $SERVER_LOCK_FILE" >&2
    exit 1
fi

# conda 환경 활성화 (bash 전용)
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    source $(conda info --base)/etc/profile.d/conda.sh
fi

conda activate pose-cuda 2>/dev/null || true

# 환경 변수 설정
export POSE_YOLO_DEVICE=0
export POSE_YOLO_SEG_WEIGHT=${POSE_YOLO_SEG_WEIGHT:-/home/dmc/AI/DMC_POSE/bed_seg/runs/bed_seg/weights/best.pt}
export POSE_PARALLEL_WORKERS=3
export POSE_INFERENCE_URGENT_QUOTA=${POSE_INFERENCE_URGENT_QUOTA:-4}
export POSE_FRAME_WIDTH=640
export POSE_ANALYSIS_ROTATION=${POSE_ANALYSIS_ROTATION:-90}
export POSE_PERSON_CONF=${POSE_PERSON_CONF:-0.03}
export POSE_REPLAY_PERSON_CONF=${POSE_REPLAY_PERSON_CONF:-0.03}
export POSE_STRONG_BOX_CONF=${POSE_STRONG_BOX_CONF:-0.5}
export POSE_STRONG_MIN_AREA_RATIO=${POSE_STRONG_MIN_AREA_RATIO:-0.016}
export POSE_WEAK_BOX_CONF=${POSE_WEAK_BOX_CONF:-0.5}
export POSE_WEAK_MIN_VISIBLE=${POSE_WEAK_MIN_VISIBLE:-8}
export POSE_WEAK_MIN_KP_MEAN=${POSE_WEAK_MIN_KP_MEAN:-0.25}
export POSE_WEAK_MIN_AREA_RATIO=${POSE_WEAK_MIN_AREA_RATIO:-0.025}
export POSE_SEG_EVERY=3
export POSE_SERVER_PORT=8000
export POSE_BED_REFINER=${POSE_BED_REFINER:-1}
export POSE_BED_REFINER_WEIGHT=${POSE_BED_REFINER_WEIGHT:-$SCRIPT_DIR/mobile_sam.pt}
export POSE_BED_REFINER_DEVICE=${POSE_BED_REFINER_DEVICE:-cpu}
export POSE_BED_REFINER_MIN_AREA_RATIO=${POSE_BED_REFINER_MIN_AREA_RATIO:-0.04}
export POSE_BED_REFINER_MAX_AREA_RATIO=${POSE_BED_REFINER_MAX_AREA_RATIO:-0.65}
export POSE_BED_REFINER_MIN_EXTENT_RATIO=${POSE_BED_REFINER_MIN_EXTENT_RATIO:-0.40}

# Neural-network independent fast motion watcher.
export POSE_MOTION_WATCHER_FPS=${POSE_MOTION_WATCHER_FPS:-20}
export POSE_MOTION_SMALL_WIDTH=${POSE_MOTION_SMALL_WIDTH:-160}
export POSE_MOTION_SMALL_HEIGHT=${POSE_MOTION_SMALL_HEIGHT:-90}
export POSE_MOTION_RATIO_THRESHOLD=${POSE_MOTION_RATIO_THRESHOLD:-0.018}
export POSE_MOTION_CONSECUTIVE_HITS=${POSE_MOTION_CONSECUTIVE_HITS:-2}
export POSE_MOTION_BURST_HOLD_SEC=${POSE_MOTION_BURST_HOLD_SEC:-3.0}
export POSE_PRE_EVENT_SECONDS=${POSE_PRE_EVENT_SECONDS:-10.0}
export POSE_PRE_EVENT_HZ=${POSE_PRE_EVENT_HZ:-20.0}
export POSE_PRE_EVENT_FRAME_WIDTH=${POSE_PRE_EVENT_FRAME_WIDTH:-640}
export POSE_PRE_EVENT_JPEG_QUALITY=${POSE_PRE_EVENT_JPEG_QUALITY:-70}
export POSE_PRE_EVENT_REPLAY=${POSE_PRE_EVENT_REPLAY:-0}
export POSE_PRE_EVENT_REPLAY_SECONDS=${POSE_PRE_EVENT_REPLAY_SECONDS:-8.0}
export POSE_PRE_EVENT_REPLAY_MAX_FRAMES=${POSE_PRE_EVENT_REPLAY_MAX_FRAMES:-150}
export POSE_PRE_EVENT_REPLAY_BATCH_SIZE=${POSE_PRE_EVENT_REPLAY_BATCH_SIZE:-8}
export POSE_PRE_EVENT_REPLAY_HOLD_SEC=${POSE_PRE_EVENT_REPLAY_HOLD_SEC:-5.0}
export POSE_PRE_EVENT_REPLAY_DEADLINE_SEC=${POSE_PRE_EVENT_REPLAY_DEADLINE_SEC:-6.0}
export POSE_EMPTY_PROBE_HZ=${POSE_EMPTY_PROBE_HZ:-0.75}
export POSE_OCCUPIED_POSE_INTERVAL_SEC=${POSE_OCCUPIED_POSE_INTERVAL_SEC:-0.09}
export POSE_LIVE_TCN_MAX_INTERVAL_SEC=${POSE_LIVE_TCN_MAX_INTERVAL_SEC:-0.25}
export POSE_TRACK_TTL_SEC=${POSE_TRACK_TTL_SEC:-5.0}
export POSE_PRIMARY_SWITCH_MARGIN=${POSE_PRIMARY_SWITCH_MARGIN:-0.25}
export POSE_SHADOW_RECORD=${POSE_SHADOW_RECORD:-1}
export POSE_SHADOW_RECORD_INTERVAL_SEC=${POSE_SHADOW_RECORD_INTERVAL_SEC:-0.5}
export POSE_SHADOW_RECORD_DIR=${POSE_SHADOW_RECORD_DIR:-$SCRIPT_DIR/runtime_data/shadow_features}
export POSE_TEMPORAL_SESSION_RECORD=${POSE_TEMPORAL_SESSION_RECORD:-1}
export POSE_TEMPORAL_SESSION_RECORD_DIR=${POSE_TEMPORAL_SESSION_RECORD_DIR:-$SCRIPT_DIR/runtime_data/temporal_sessions}
export POSE_TEMPORAL_SESSION_CAMERAS=${POSE_TEMPORAL_SESSION_CAMERAS:-bed_161}
export POSE_TEMPORAL_SESSION_PRE_SEC=${POSE_TEMPORAL_SESSION_PRE_SEC:-10.0}
export POSE_TEMPORAL_SESSION_POST_SEC=${POSE_TEMPORAL_SESSION_POST_SEC:-10.0}
export POSE_TEMPORAL_SESSION_MAX_SEC=${POSE_TEMPORAL_SESSION_MAX_SEC:-180.0}
export POSE_TEMPORAL_SESSION_MODEL_REARM_SEC=${POSE_TEMPORAL_SESSION_MODEL_REARM_SEC:-60.0}
# Authenticated Pi results are scheduling hints. Central YOLO11m remains authoritative.
export POSE_EDGE_SIGNAL=${POSE_EDGE_SIGNAL:-1}
export POSE_EDGE_WAKE_SCHEDULER=${POSE_EDGE_WAKE_SCHEDULER:-1}
export POSE_EDGE_SIGNAL_URL=${POSE_EDGE_SIGNAL_URL:-http://127.0.0.1:8020}
export POSE_EDGE_SIGNAL_TOKEN_FILE=${POSE_EDGE_SIGNAL_TOKEN_FILE:-$SCRIPT_DIR/runtime_data/edge_control/api_token}
export POSE_EDGE_SIGNAL_MAX_AGE_SEC=${POSE_EDGE_SIGNAL_MAX_AGE_SEC:-4.0}
export POSE_EDGE_MANAGED_CAMERAS=${POSE_EDGE_MANAGED_CAMERAS:-bed_161}
export POSE_EDGE_FAILOVER_GRACE_SEC=${POSE_EDGE_FAILOVER_GRACE_SEC:-3.0}
export POSE_EDGE_MANAGED_EMPTY_PROBE_HZ=${POSE_EDGE_MANAGED_EMPTY_PROBE_HZ:-0.05}
# 카메라별 침대 영역 자동 검출/합의/캐시. 수동 ROI는 사용하지 않음.
# Cache namespace includes the Bed-Seg input contract. Older rot90 caches were
# learned by incorrectly segmenting an already-rotated frame and must never be
# restored as if they used native-camera segmentation.
export POSE_AUTO_BED_CACHE_DIR=${POSE_AUTO_BED_CACHE_DIR:-/home/dmc/AI/DMC_POSE/bed_roi/auto_cache_sam_multipoint_rot${POSE_ANALYSIS_ROTATION}}
export POSE_AUTO_BED_WINDOW=${POSE_AUTO_BED_WINDOW:-5}
export POSE_AUTO_BED_MIN_DETECTIONS=${POSE_AUTO_BED_MIN_DETECTIONS:-3}
export POSE_AUTO_BED_CONSENSUS_IOU=${POSE_AUTO_BED_CONSENSUS_IOU:-0.75}
export POSE_AUTO_BED_REFRESH_SEC=${POSE_AUTO_BED_REFRESH_SEC:-300}

# 중앙 서버 TCN shadow 추론 (실제 경보에는 아직 연결하지 않음)
export POSE_TCN_SHADOW=${POSE_TCN_SHADOW:-1}
export POSE_TCN_MODEL=${POSE_TCN_MODEL:-/home/dmc/AI/DMC_POSE/runs/temporal_tcn/gmdcsa24_tcn/model.pt}
export POSE_TCN_REPORT=${POSE_TCN_REPORT:-/home/dmc/AI/DMC_POSE/runs/temporal_tcn/gmdcsa24_tcn/report.json}
export POSE_TCN_DEVICE=${POSE_TCN_DEVICE:-cpu}

# DMC_POSE 디렉토리로 이동
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════╗"
echo "║   🎥 Multi-Camera Parallel GPU Pipeline Server    ║"
echo "║   모든 카메라 동시 분석 (6대)                    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "⚙️  설정:"
echo "   - YOLO Device: $POSE_YOLO_DEVICE"
echo "   - Bed Seg Weight: $POSE_YOLO_SEG_WEIGHT"
echo "   - Bed Refiner: $POSE_BED_REFINER / $POSE_BED_REFINER_WEIGHT ($POSE_BED_REFINER_DEVICE, ROI refresh only)"
echo "   - Person Pose Confidence: $POSE_PERSON_CONF"
echo "   - Replay Pose Confidence: $POSE_REPLAY_PERSON_CONF"
echo "   - Strong Pose Gate: box>=$POSE_STRONG_BOX_CONF, area>=$POSE_STRONG_MIN_AREA_RATIO"
echo "   - Weak Pose Gate: box>=$POSE_WEAK_BOX_CONF, visible>=$POSE_WEAK_MIN_VISIBLE, kp_mean>=$POSE_WEAK_MIN_KP_MEAN, area>=$POSE_WEAK_MIN_AREA_RATIO"
echo "   - Model Host Workers: $POSE_PARALLEL_WORKERS"
echo "   - Central Scheduler: latest-only, urgent quota $POSE_INFERENCE_URGENT_QUOTA"
echo "   - Frame Width: $POSE_FRAME_WIDTH"
echo "   - Analysis Rotation: ${POSE_ANALYSIS_ROTATION}° clockwise (viewer remains raw)"
echo "   - Server Port: $POSE_SERVER_PORT"
echo "   - Motion Watcher: ${POSE_MOTION_WATCHER_FPS} FPS, ${POSE_MOTION_SMALL_WIDTH}x${POSE_MOTION_SMALL_HEIGHT}"
echo "   - Motion Burst: ratio ${POSE_MOTION_RATIO_THRESHOLD}, ${POSE_MOTION_CONSECUTIVE_HITS} hits, hold ${POSE_MOTION_BURST_HOLD_SEC}s"
echo "   - Pre-event Ring: ${POSE_PRE_EVENT_SECONDS}s @ ${POSE_PRE_EVENT_HZ}Hz, ${POSE_PRE_EVENT_FRAME_WIDTH}px JPEG q${POSE_PRE_EVENT_JPEG_QUALITY}"
echo "   - Catch-up Replay: ${POSE_PRE_EVENT_REPLAY}, async single-flight P3 + ${POSE_PRE_EVENT_SECONDS}s cooldown, ${POSE_PRE_EVENT_REPLAY_SECONDS}s/${POSE_PRE_EVENT_REPLAY_MAX_FRAMES} frames, batch ${POSE_PRE_EVENT_REPLAY_BATCH_SIZE}, budget ${POSE_PRE_EVENT_REPLAY_DEADLINE_SEC}s"
echo "   - Empty Pose Probe: ${POSE_EMPTY_PROBE_HZ} Hz"
echo "   - Occupied Pose Cadence: ${POSE_OCCUPIED_POSE_INTERVAL_SEC}s capture-time pacing"
echo "   - Live TCN Cadence: observed-only 70-${POSE_LIVE_TCN_MAX_INTERVAL_SEC}s max-gap"
echo "   - Person Tracking: TTL ${POSE_TRACK_TTL_SEC}s, switch margin ${POSE_PRIMARY_SWITCH_MARGIN}"
echo "   - Feature Recorder: ${POSE_SHADOW_RECORD} (${POSE_SHADOW_RECORD_INTERVAL_SEC}s, feature-only)"
echo "   - Temporal Sessions: ${POSE_TEMPORAL_SESSION_RECORD} (${POSE_TEMPORAL_SESSION_CAMERAS}, pre ${POSE_TEMPORAL_SESSION_PRE_SEC}s + post ${POSE_TEMPORAL_SESSION_POST_SEC}s, 109D only)"
echo "   - Temporal Model-only Rearm: ${POSE_TEMPORAL_SESSION_MODEL_REARM_SEC}s (Edge/person/Fusion never suppressed)"
echo "   - Edge Wake Bridge: ${POSE_EDGE_SIGNAL} (scheduler=${POSE_EDGE_WAKE_SCHEDULER}, max-age=${POSE_EDGE_SIGNAL_MAX_AGE_SEC}s)"
echo "   - Edge Managed: ${POSE_EDGE_MANAGED_CAMERAS} (fallback=${POSE_EDGE_FAILOVER_GRACE_SEC}s, empty-probe=${POSE_EDGE_MANAGED_EMPTY_PROBE_HZ}Hz)"
echo "   - Auto Bed ROI: consensus $POSE_AUTO_BED_MIN_DETECTIONS/$POSE_AUTO_BED_WINDOW, IoU $POSE_AUTO_BED_CONSENSUS_IOU"
echo "   - Bed Seg Refresh: ${POSE_AUTO_BED_REFRESH_SEC}s"
echo "   - TCN Shadow: $POSE_TCN_SHADOW ($POSE_TCN_DEVICE)"
echo ""
echo "🌐 접속:"
echo "   - Viewer: http://localhost:8000/viewer"
echo "   - API: http://localhost:8000/status"
echo ""
echo "📁 작업 디렉토리: $(pwd)"
echo ""

# 모델 파일 확인
echo "✓ 모델 파일 확인:"
[[ -f "$POSE_YOLO_SEG_WEIGHT" ]] && echo "  ✅ Bed Seg: $POSE_YOLO_SEG_WEIGHT" || echo "  ❌ Bed Seg weight (없음): $POSE_YOLO_SEG_WEIGHT"
[[ "$POSE_BED_REFINER" != "1" || -f "$POSE_BED_REFINER_WEIGHT" ]] && echo "  ✅ Bed Refiner: $POSE_BED_REFINER_WEIGHT" || echo "  ❌ Bed Refiner weight (없음): $POSE_BED_REFINER_WEIGHT"
[[ -f "yolo11m-pose.pt" ]] && echo "  ✅ yolo11m-pose.pt" || echo "  ❌ yolo11m-pose.pt (없음)"
[[ -f "my_model_six_check.keras" ]] && echo "  ✅ my_model_six_check.keras" || echo "  ❌ my_model_six_check.keras (없음)"
[[ -f "$POSE_TCN_MODEL" ]] && echo "  ✅ TCN model.pt" || echo "  ⚠️ TCN model.pt (shadow 비활성화 예정)"
[[ -f "$POSE_TCN_REPORT" ]] && echo "  ✅ TCN report.json" || echo "  ⚠️ TCN report.json (shadow 비활성화 예정)"
echo ""

python server_all_cameras.py
