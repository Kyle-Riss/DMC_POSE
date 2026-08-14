"""
GPU 병렬 추론 서버 - RTSP + 병렬 YOLO seg/pose + Keras 6-class
병렬화: ThreadPoolExecutor로 seg와 pose를 동시에 실행
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
from threading import Lock, Thread, Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from queue import Queue
import threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# bed_monitor 임포트 제거 (없는 모듈)
# from bed_monitor.bed_zone import build_approx_bed_zone
# from bed_monitor.config import load_preset
# from bed_monitor.features import MotionState
# from bed_monitor.live import apply_fall_scoring, enrich_from_keypoints
# from bed_monitor.scoring import FallScorer
# from bed_monitor.temporal import LiveEventTracker
# from bed_roi.roi_utils import apply_bed_roi, load_bed_roi
# from rail.rail_detect import detect_both_rails, draw_rail_rois, load_rail_config

# Keras only on CPU; YOLO uses PyTorch GPU
tf.config.set_visible_devices([], 'GPU')
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ── RTSP 카메라 설정 ──────────────────────────────────
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

# ── 설정 ──────────────────────────────────────────────
YOLO_SEG_WEIGHT = 'yolo11n-bed-seg.pt'
YOLO_SEG_CLASS = 0
YOLO_POSE_WEIGHT = 'yolo11m-pose.pt'
YOLO_DEVICE = os.environ.get('POSE_YOLO_DEVICE', '0')
SEG_EVERY_N = max(1, int(os.environ.get('POSE_SEG_EVERY', '3')))

# 카메라 선택 (환경 변수 또는 기본값)
CAMERA_ID = os.environ.get('POSE_CAMERA_ID', 'raspi_bed_001')
CAMERA_CONFIG = CAMERA_CONFIGS.get(CAMERA_ID, CAMERA_CONFIGS['raspi_bed_001'])
RTSP_URL = os.environ.get('POSE_RTSP_URL', CAMERA_CONFIG['rtsp_url'])
FRAME_WIDTH = int(os.environ.get('POSE_FRAME_WIDTH', '640'))
VIEWER_SCALE = float(os.environ.get('POSE_VIEWER_SCALE', '1.5'))
HUD_FONT = float(os.environ.get('POSE_HUD_FONT', '0.38'))
HUD_FONT_SM = float(os.environ.get('POSE_HUD_FONT_SM', '0.32'))
YOLO_PLOT_LINE_WIDTH = int(os.environ.get('POSE_YOLO_LINE_WIDTH', '1'))
YOLO_PLOT_KPT_RADIUS = int(os.environ.get('POSE_YOLO_KPT_RADIUS', '2'))
YOLO_PLOT_LABELS = os.environ.get('POSE_YOLO_LABELS', '0') == '1'
YOLO_PLOT_BOXES = os.environ.get('POSE_YOLO_BOXES', '0') == '1'

POSE_KERAS_MODEL = 'my_model_six_check.keras'
PRESET_PATH = os.environ.get('POSE_PRESET_PATH', 'config/default.json')
BED_ROI_PATH = os.environ.get('POSE_BED_ROI', 'bed_roi/bed_roi.json')
USE_BED_ROI = os.environ.get('POSE_USE_BED_ROI', '1') == '1'
BED_SEG_CONF = float(os.environ.get('POSE_BED_SEG_CONF', '0.1'))

# 병렬 처리 설정
PARALLEL_WORKERS = int(os.environ.get('POSE_PARALLEL_WORKERS', '2'))
FRAME_QUEUE_SIZE = int(os.environ.get('POSE_FRAME_QUEUE_SIZE', '30'))

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
        """
        Seg와 Pose를 병렬로 실행
        Returns: (seg_result, pose_result)
        """
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

# ── 헬퍼 함수들 (기존과 동일) ──────────────────────────────
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
    bed = {'mask': None, 'bbox': None, 'source': 'none', 'zone_quality': 'none'}
    
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

def _bbox_from_boxes(seg_result, h: int, w: int) -> tuple:
    """박스 추출"""
    if seg_result is None or seg_result.boxes is None or len(seg_result.boxes) == 0:
        return None
    best_idx = int(seg_result.boxes.conf.argmax())
    x1, y1, x2, y2 = seg_result.boxes.xyxy[best_idx].cpu().numpy()
    return (int(x1), int(y1), int(x2), int(y2))

# ── 서버 상태 ──────────────────────────────────────────────
class AnalysisState(BaseModel):
    in_bed: str = "NO"
    pose: str = "None"
    pose_conf: float = 0.0
    timestamp: str = ""
    latency_ms: float = 0.0
    frame_age_ms: float = 0.0
    pipeline_fps: float = 0.0

state_lock = Lock()
current_state = AnalysisState()
frame_buffer = None
analysis_running = False
last_frame_time = time.time()
frame_times = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 생명주기"""
    global analysis_running, inference_pool
    
    logging.info(f"[INFO] YOLO GPU={YOLO_DEVICE} | parallel workers={PARALLEL_WORKERS}")
    logging.info(f"[INFO] seg every {SEG_EVERY_N} frames | RTSP={RTSP_URL}")
    
    # 모델 로드
    inference_pool = ParallelInferencePool(
        YOLO_SEG_WEIGHT, YOLO_POSE_WEIGHT,
        device=YOLO_DEVICE, workers=PARALLEL_WORKERS
    )
    keras_clf = keras.models.load_model(POSE_KERAS_MODEL)
    preset = load_preset(PRESET_PATH)
    event_tracker = LiveEventTracker(preset.get("event_config", {}))
    
    analysis_running = True
    analysis_thread = Thread(
        target=run_analysis,
        args=(inference_pool, keras_clf, preset, event_tracker),
        daemon=True
    )
    analysis_thread.start()
    
    yield
    
    analysis_running = False
    inference_pool.shutdown()
    analysis_thread.join(timeout=5)

app = FastAPI(lifespan=lifespan)
inference_pool = None

def run_analysis(inference_pool, keras_clf, preset, event_tracker):
    """병렬 추론 루프"""
    global current_state, frame_buffer, last_frame_time, frame_times
    
    logging.info(f"[INFO] 카메라 선택: {CAMERA_ID} ({CAMERA_CONFIG['name']})")
    logging.info(f"[INFO] RTSP: {RTSP_URL}")
    
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logging.error(f"[ERROR] RTSP 열기 실패: {RTSP_URL}")
        return
    
    frame_idx = 0
    cached_bed = None
    motion_state = MotionState()
    fall_scorer = FallScorer(preset.get("scoring", {}))
    
    frame_capture_time = None
    
    while analysis_running:
        ret, frame = cap.read()
        if not ret:
            logging.warning("[WARNING] RTSP 프레임 읽기 실패, 재연결 시도...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            continue
        
        frame_capture_time = time.time()
        frame = imutils.resize(frame, width=FRAME_WIDTH)
        fh, fw = frame.shape[:2]
        frame_idx += 1
        
        infer_start = time.time()
        
        # ★ 병렬 추론: seg와 pose를 동시에 실행
        run_seg = (frame_idx % SEG_EVERY_N == 1 or cached_bed is None)
        seg_res, pose_res = inference_pool.infer_parallel(
            frame, run_seg=run_seg, run_pose=True
        )
        
        # Segmentation 결과 처리
        if seg_res is not None:
            fresh_bed = extract_bed_detection(seg_res[0], fh, fw)
            if fresh_bed.get('bbox') is not None or fresh_bed.get('mask') is not None:
                cached_bed = fresh_bed
        
        bed = cached_bed or {'mask': None, 'bbox': None, 'source': 'none'}
        
        # ROI 적용
        roi_bbox = load_bed_roi(Path(BED_ROI_PATH), fw, fh) if USE_BED_ROI else None
        if roi_bbox is not None:
            bed = apply_bed_roi(bed, roi_bbox, fh, fw)
        
        if preset.get("bed_zone"):
            bed = build_approx_bed_zone(bed, roi_bbox, fh, fw, preset)
        
        # Pose 결과 처리
        in_bed = "NO"
        pose_display = "None"
        pose_conf = 0.0
        person_detected = False
        
        if pose_res is not None and len(pose_res) > 0 and len(pose_res[0].keypoints) > 0:
            person_detected = True
            kp = pose_res[0].keypoints[0]
            kpt_xy = kp.xy[0] if hasattr(kp, 'xy') else None
            
            if kpt_xy is not None and bed.get('mask') is not None:
                mask = bed['mask']
                kx, ky = int(kpt_xy[0]), int(kpt_xy[1])
                if 0 <= ky < mask.shape[0] and 0 <= kx < mask.shape[1]:
                    in_bed = "YES" if mask[ky, kx] > 0 else "NO"
            
            # 6-class 분류
            if len(kp.xy) >= 12:
                kpts_input = kp.xy.flatten().reshape(1, -1)
                pred = keras_clf.predict(kpts_input, verbose=0)
                pose_idx = np.argmax(pred[0])
                pose_conf = float(pred[0][pose_idx])
                poses_names = ['front_lying', 'prone_back', 'side_near', 'side_far', 'sitting_center', 'sitting_edge']
                pose_display = poses_names[pose_idx] if pose_idx < len(poses_names) else 'Unknown'
        
        infer_time = (time.time() - infer_start) * 1000
        
        # 상태 업데이트
        with state_lock:
            current_state.in_bed = in_bed
            current_state.pose = pose_display
            current_state.pose_conf = pose_conf
            # Use UTC timestamp for monitoring
            current_state.timestamp = datetime.utcnow().isoformat() + 'Z'
            current_state.latency_ms = infer_time
            current_state.frame_age_ms = (time.time() - frame_capture_time) * 1000
            
            # FPS 계산
            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) > 1:
                fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                current_state.pipeline_fps = fps
    
    cap.release()

@app.get("/status")
def get_status():
    """현재 분석 상태"""
    with state_lock:
        return current_state.model_dump()

@app.get("/health")
def health():
    """헬스 체크"""
    return {
        "server": "ok",
        "analysis_running": analysis_running
    }

@app.get("/viewer", response_class=HTMLResponse)
def viewer():
    """HTML 뷰어"""
    return """
    <html>
    <head>
        <title>Bed Monitor - Parallel GPU Pipeline</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            .status { font-size: 18px; padding: 10px; border: 1px solid #ccc; margin: 10px 0; }
            .label { font-weight: bold; }
            .value { color: #0066cc; }
        </style>
        <script>
            async function updateStatus() {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    document.getElementById('status').innerHTML = `
                        <div class="status">
                            <span class="label">In Bed:</span> <span class="value">${data.in_bed}</span>
                        </div>
                        <div class="status">
                            <span class="label">Pose:</span> <span class="value">${data.pose} (${(data.pose_conf * 100).toFixed(1)}%)</span>
                        </div>
                        <div class="status">
                            <span class="label">Pipeline FPS:</span> <span class="value">${data.pipeline_fps.toFixed(1)}</span>
                        </div>
                        <div class="status">
                            <span class="label">Latency:</span> <span class="value">${data.latency_ms.toFixed(1)}ms</span>
                        </div>
                        <div class="status">
                            <span class="label">Timestamp:</span> <span class="value">${data.timestamp}</span>
                        </div>
                    `;
                } catch(e) {
                    console.error(e);
                }
            }
            setInterval(updateStatus, 500);
        </script>
    </head>
    <body>
        <h1>🛏️ Bed Monitor - Parallel GPU Pipeline</h1>
        <p>⚡ <strong>GPU 병렬 추론 모드</strong> - Seg와 Pose를 동시 실행</p>
        <div id="status">Loading...</div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
