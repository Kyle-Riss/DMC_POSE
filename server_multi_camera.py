"""
4 Camera Parallel GPU Inference Server
RTSP + 병렬 seg/pose + Keras 6-class (4개 카메라 동시)
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import cv2
import numpy as np
import tensorflow as tf
import keras
import logging
import imutils
import time
from pathlib import Path
from ultralytics import YOLO
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Keras only on CPU; YOLO uses PyTorch GPU
tf.config.set_visible_devices([], 'GPU')
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ── 4 카메라 설정 ──────────────────────────────────────────────
CAMERA_CONFIGS = {
    'raspi_bed_001': {
        'name': 'Bed 1 (Main)',
        'rtsp_url': 'rtsp://192.168.0.161:8554/stream',
        'room': 'room_161'
    },
    'raspi_bed_002': {
        'name': 'Bed 2 (Secondary)',
        'rtsp_url': 'rtsp://192.168.0.174:8554/stream',
        'room': 'room_174'
    },
    'raspi_bed_003': {
        'name': 'Bed 3 (Testing)',
        'rtsp_url': 'rtsp://192.168.0.178:8554/stream',
        'room': 'room_178'
    },
    'raspi_bed_004': {
        'name': 'Bed 4 (Backup)',
        'rtsp_url': 'rtsp://192.168.0.179:8554/stream',
        'room': 'room_179'
    },
}

# ── 글로벌 설정 ──────────────────────────────────────────────
YOLO_SEG_WEIGHT = 'yolo11n-bed-seg.pt'
YOLO_SEG_CLASS = 0
YOLO_POSE_WEIGHT = 'yolo11m-pose.pt'
YOLO_DEVICE = os.environ.get('POSE_YOLO_DEVICE', '0')
SEG_EVERY_N = max(1, int(os.environ.get('POSE_SEG_EVERY', '3')))
FRAME_WIDTH = int(os.environ.get('POSE_FRAME_WIDTH', '640'))
BED_SEG_CONF = float(os.environ.get('POSE_BED_SEG_CONF', '0.1'))
PARALLEL_WORKERS = int(os.environ.get('POSE_PARALLEL_WORKERS', '2'))
POSE_KERAS_MODEL = 'my_model_six_check.keras'
BED_ROI_PATH = os.environ.get('POSE_BED_ROI', 'bed_roi/bed_roi.json')
USE_BED_ROI = os.environ.get('POSE_USE_BED_ROI', '1') == '1'

class ParallelInferencePool:
    """YOLO seg/pose 병렬 추론 풀"""
    def __init__(self, seg_weight, pose_weight, device='0', workers=2):
        self.seg_model = YOLO(seg_weight)
        self.pose_model = YOLO(pose_weight)
        self.device = device
        self.workers = workers
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.lock = Lock()
        
    def infer_seg(self, frame, conf=0.1, classes=[0]):
        """Segmentation 추론"""
        with self.lock:
            result = self.seg_model.predict(
                frame,
                conf=conf,
                classes=classes,
                device=self.device,
                verbose=False
            )
        return result
    
    def infer_pose(self, frame, conf=0.5):
        """Pose 추론"""
        with self.lock:
            result = self.pose_model.predict(
                frame,
                conf=conf,
                device=self.device,
                verbose=False
            )
        return result
    
    def infer_parallel(self, frame, run_seg=True, run_pose=True):
        """Seg와 Pose를 병렬로 실행"""
        futures = {}
        
        if run_seg:
            futures['seg'] = self.executor.submit(
                self.infer_seg, frame, conf=BED_SEG_CONF, classes=[YOLO_SEG_CLASS]
            )
        if run_pose:
            futures['pose'] = self.executor.submit(
                self.infer_pose, frame, conf=0.5
            )
        
        results = {}
        for key, future in futures.items():
            try:
                results[key] = future.result(timeout=5.0)
            except Exception as e:
                logging.error(f"Inference error ({key}): {e}")
                results[key] = None
        
        return results.get('seg'), results.get('pose')
    
    def shutdown(self):
        self.executor.shutdown(wait=True)

# ── 헬퍼 함수 ──────────────────────────────────────────────
def _mask_to_frame(mask_tensor, h: int, w: int) -> np.ndarray:
    """마스크 텐서 → 프레임 좌표"""
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[0]
    mask_np = mask_tensor.cpu().numpy()
    if mask_np.shape != (h, w):
        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
    return (mask_np * 255).astype(np.uint8)

def extract_bed_detection(seg_result, h: int, w: int) -> dict:
    """침대 검출 추출"""
    bed = {'mask': None, 'bbox': None, 'source': 'none'}
    
    if seg_result is None or seg_result.boxes is None or len(seg_result.boxes) == 0:
        return bed
    
    best_idx = int(seg_result.boxes.conf.argmax())
    
    if seg_result.masks is not None and len(seg_result.masks) > best_idx:
        m_np = _mask_to_frame(seg_result.masks.data[best_idx], h, w)
        bed['mask'] = m_np
        bed['source'] = 'seg_mask'
    else:
        x1, y1, x2, y2 = seg_result.boxes.xyxy[best_idx].cpu().numpy()
        bed['bbox'] = (int(x1), int(y1), int(x2), int(y2))
        bed['source'] = 'seg_bbox'
    
    return bed

class AnalysisState(BaseModel):
    """카메라별 분석 상태"""
    camera_id: str = "none"
    camera_name: str = "none"
    in_bed: str = "NO"
    pose: str = "None"
    pose_conf: float = 0.0
    timestamp: str = ""
    latency_ms: float = 0.0
    pipeline_fps: float = 0.0
    status: str = "disconnected"
    fall_score: float = 0.0
    fall_level: str = "SAFE"
    fall_status: str = "NO_PERSON"

# ── 글로벌 상태 ──────────────────────────────────────────────
state_lock = Lock()
camera_states = {camera_id: AnalysisState(camera_id=camera_id) 
                 for camera_id in CAMERA_CONFIGS.keys()}
analysis_running = False
inference_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 생명주기"""
    global analysis_running, inference_pool
    
    logging.info(f"[INFO] 🚀 4 Camera Parallel Pipeline 시작")
    logging.info(f"[INFO] YOLO GPU={YOLO_DEVICE} | Parallel Workers={PARALLEL_WORKERS}")
    
    # 모델 로드
    inference_pool = ParallelInferencePool(
        YOLO_SEG_WEIGHT, YOLO_POSE_WEIGHT,
        device=YOLO_DEVICE, workers=PARALLEL_WORKERS
    )
    keras_clf = keras.models.load_model(POSE_KERAS_MODEL)
    
    analysis_running = True
    
    # 각 카메라별로 분석 스레드 시작
    threads = []
    for camera_id in CAMERA_CONFIGS.keys():
        thread = Thread(
            target=run_analysis,
            args=(camera_id, inference_pool, keras_clf),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    yield
    
    analysis_running = False
    inference_pool.shutdown()
    for thread in threads:
        thread.join(timeout=5)

app = FastAPI(lifespan=lifespan)

def run_analysis(camera_id: str, inference_pool, keras_clf):
    """카메라별 분석 루프"""
    global camera_states, analysis_running
    
    camera_config = CAMERA_CONFIGS[camera_id]
    rtsp_url = camera_config['rtsp_url']
    
    logging.info(f"[🎥 {camera_id}] {camera_config['name']} 분석 시작")
    logging.info(f"[🎥 {camera_id}] RTSP: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logging.error(f"[❌ {camera_id}] RTSP 열기 실패")
        with state_lock:
            camera_states[camera_id].status = "disconnected"
        return
    
    frame_idx = 0
    cached_bed = None
    frame_times = []
    poses_names = ['front_lying', 'prone_back', 'side_near', 'side_far', 'sitting_center', 'sitting_edge']
    
    while analysis_running:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"[⚠️  {camera_id}] 프레임 읽기 실패, 재연결 시도...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url)
            continue
        
        frame_capture_time = time.time()
        frame = imutils.resize(frame, width=FRAME_WIDTH)
        fh, fw = frame.shape[:2]
        frame_idx += 1
        
        infer_start = time.time()
        
        # ★ 병렬 추론
        run_seg = (frame_idx % SEG_EVERY_N == 1 or cached_bed is None)
        seg_res, pose_res = inference_pool.infer_parallel(
            frame, run_seg=run_seg, run_pose=True
        )
        
        # Segmentation 처리
        if seg_res is not None:
            fresh_bed = extract_bed_detection(seg_res[0], fh, fw)
            if fresh_bed.get('bbox') is not None or fresh_bed.get('mask') is not None:
                cached_bed = fresh_bed
        
        bed = cached_bed or {'mask': None, 'bbox': None, 'source': 'none'}
        
        # [RULE] ROI 적용
        from bed_roi.roi_utils import apply_bed_roi, load_bed_roi
        roi_bbox = load_bed_roi(Path(BED_ROI_PATH), fw, fh) if USE_BED_ROI else None
        if roi_bbox is not None:
            bed = apply_bed_roi(bed, roi_bbox, fh, fw)
        
        # [RULE] Feature Enrichment & Event Tracking
        from bed_monitor.live import enrich_from_keypoints
        from bed_monitor.features import MotionState
        from bed_monitor.temporal import LiveEventTracker
        from bed_monitor.scoring import FallScorer, apply_fall_scoring
        from bed_monitor.rail import detect_both_rails, load_rail_config # 가상 모듈
        
        # 카메라별 상태 추적을 위한 초기화
        if not hasattr(run_analysis, 'motion_state'):
            run_analysis.motion_state = MotionState()
        if not hasattr(run_analysis, 'event_tracker'):
            preset = {
                "risk_thresholds": {"overflow_high": 0.25, "overflow_med": 0.15, "overflow_low": 0.05},
                "motion": {"ema_alpha": 0.3, "hold_sec": 0.5},
                "events": {"min_torso_kpts": 2, "min_valid_kpts": 5, "attach_ratio_min": 0.35},
                "inference": {"kpt_conf": 0.3, "skeleton_min_core_kpts": 2, "skeleton_min_total_kpts": 5},
                "scoring": {"enabled": True}
            }
            run_analysis.event_tracker = LiveEventTracker(preset)
            run_analysis.fall_scorer = FallScorer(preset.get("scoring", {}))
            run_analysis.preset = preset
        
        # Pose 처리
        in_bed = "NO"
        pose_display = "None"
        pose_conf = 0.0
        person_detected = False
        fall_score = 0.0
        fall_level = "SAFE"
        fall_status = "NO_PERSON"
        
        if pose_res is not None and len(pose_res) > 0 and len(pose_res[0].keypoints) > 0:
            kp = pose_res[0].keypoints[0]
            kpts_xy = kp.xy[0].cpu().numpy() if hasattr(kp, 'xy') else None
            kpts_conf = kp.conf[0].cpu().numpy() if hasattr(kp, 'conf') and kp.conf is not None else np.ones(17)
            
            if kpts_xy is not None:
                t_sec = time.time()
                # 1. Feature Enrichment
                feat = enrich_from_keypoints(kpts_xy, kpts_conf, bed, run_analysis.motion_state, t_sec, run_analysis.preset)
                # 2. Event Tracking
                run_analysis.event_tracker.update(t_sec, feat)
                
                person_detected = bool(feat["person_detected"])
                in_bed = "YES" if feat["in_bed"] else "NO"
                
                # 3. Fall Scoring
                apply_fall_scoring(
                    feat, kpts_xy, kpts_conf, bed.get("bbox"), 
                    run_analysis.preset, run_analysis.fall_scorer
                )
                fall_score = float(feat.get("fall_score", 0.0))
                fall_level = str(feat.get("fall_level", "SAFE"))
                fall_status = str(feat.get("fall_status", "NO_PERSON"))
            
            # 6-class 분류
            if hasattr(kp, 'xy') and len(kp.xy[0]) >= 12:
                kpts_input = kp.xy[0].cpu().numpy().flatten().reshape(1, -1)
                pred = keras_clf.predict(kpts_input, verbose=0)
                pose_idx = np.argmax(pred[0])
                pose_conf = float(pred[0][pose_idx])
                pose_display = poses_names[pose_idx] if pose_idx < len(poses_names) else 'Unknown'
        
        # 상태 업데이트
        with state_lock:
            camera_states[camera_id].in_bed = in_bed
            camera_states[camera_id].pose = pose_display
            camera_states[camera_id].pose_conf = pose_conf
            camera_states[camera_id].fall_score = fall_score
            camera_states[camera_id].fall_level = fall_level
            camera_states[camera_id].fall_status = fall_status
            # Standardize timestamps to UTC Z format
            camera_states[camera_id].timestamp = datetime.utcnow().isoformat() + 'Z'
            camera_states[camera_id].latency_ms = infer_time
            camera_states[camera_id].status = "analyzing" if person_detected else "idle"
            
            # FPS 계산
            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) > 1:
                fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                camera_states[camera_id].pipeline_fps = fps
    
    cap.release()
    logging.info(f"[✅ {camera_id}] 분석 종료")

# ── API 엔드포인트 ──────────────────────────────────────────────

@app.get("/status")
def get_status():
    """모든 카메라의 상태"""
    with state_lock:
        return {camera_id: state.model_dump() 
                for camera_id, state in camera_states.items()}

@app.get("/status/{camera_id}")
def get_camera_status(camera_id: str):
    """특정 카메라의 상태"""
    if camera_id not in camera_states:
        return {"error": f"Unknown camera: {camera_id}"}
    
    with state_lock:
        return camera_states[camera_id].model_dump()

@app.get("/health")
def health():
    """헬스 체크"""
    return {
        "server": "ok",
        "analysis_running": analysis_running,
        "total_cameras": len(CAMERA_CONFIGS),
        "cameras": list(CAMERA_CONFIGS.keys())
    }

@app.get("/viewer", response_class=HTMLResponse)
def viewer():
    """HTML 대시보드"""
    camera_html = ""
    for camera_id, config in CAMERA_CONFIGS.items():
        camera_html += f"""
        <div class="camera-card">
            <h3>{config['name']}</h3>
            <div class="status-row">
                <span class="label">In Bed:</span> <span class="value" id="inbed-{camera_id}">--</span>
            </div>
            <div class="status-row">
                <span class="label">Pose:</span> <span class="value" id="pose-{camera_id}">--</span>
            </div>
            <div class="status-row">
                <span class="label">Confidence:</span> <span class="value" id="conf-{camera_id}">--</span>
            </div>
            <div class="status-row">
                <span class="label">FPS:</span> <span class="value" id="fps-{camera_id}">--</span>
            </div>
            <div class="status-row">
                <span class="label">Latency:</span> <span class="value" id="latency-{camera_id}">--</span>
            </div>
            <div class="status-row">
                <span class="label">Status:</span> <span class="value" id="status-{camera_id}">--</span>
            </div>
        </div>
        """
    
    return f"""
    <html>
    <head>
        <title>4 Camera Parallel Monitoring</title>
        <style>
            body {{ 
                font-family: Arial; 
                margin: 20px; 
                background: #f5f5f5;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .container {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }}
            .camera-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .camera-card h3 {{
                margin-top: 0;
                color: #333;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 10px;
            }}
            .status-row {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}
            .label {{
                font-weight: bold;
                color: #555;
            }}
            .value {{
                color: #0066cc;
                font-family: monospace;
            }}
            .status-analyzing {{
                color: #00aa00;
                font-weight: bold;
            }}
            .status-idle {{
                color: #ff9900;
            }}
            .status-disconnected {{
                color: #ff0000;
            }}
        </style>
        <script>
            async function updateStatus() {{
                try {{
                    const res = await fetch('/status');
                    const data = await res.json();
                    
                    for (const [cameraId, state] of Object.entries(data)) {{
                        document.getElementById(`inbed-${{cameraId}}`).textContent = state.in_bed;
                        document.getElementById(`pose-${{cameraId}}`).textContent = state.pose;
                        document.getElementById(`conf-${{cameraId}}`).textContent = (state.pose_conf * 100).toFixed(0) + '%';
                        document.getElementById(`fps-${{cameraId}}`).textContent = state.pipeline_fps.toFixed(1);
                        document.getElementById(`latency-${{cameraId}}`).textContent = state.latency_ms.toFixed(1) + 'ms';
                        
                        const statusEl = document.getElementById(`status-${{cameraId}}`);
                        statusEl.textContent = state.status;
                        statusEl.className = `value status-${{state.status}}`;
                    }}
                }} catch(e) {{
                    console.error(e);
                }}
            }}
            setInterval(updateStatus, 500);
            updateStatus();  // 초기 호출
        </script>
    </head>
    <body>
        <div class="header">
            <h1>🛏️ 4 Camera Parallel Monitoring</h1>
            <p>⚡ <strong>GPU 병렬 추론 모드</strong> - 4개 카메라 동시 분석</p>
        </div>
        <div class="container">
            {camera_html}
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('POSE_SERVER_PORT', '8000'))
    logging.info(f"[INFO] 📡 서버 시작: http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
