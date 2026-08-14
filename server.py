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
from datetime import datetime
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from bed_monitor.bed_zone import build_approx_bed_zone
from bed_monitor.config import load_preset
from bed_monitor.features import MotionState
from bed_monitor.live import apply_fall_scoring, enrich_from_keypoints
from bed_monitor.scoring import FallScorer
from bed_monitor.temporal import LiveEventTracker
from bed_roi.roi_utils import apply_bed_roi, load_bed_roi
from rail.rail_detect import detect_both_rails, draw_rail_rois, load_rail_config

# Keras only on CPU (RTX 50xx TF GPU issue); YOLO uses PyTorch GPU separately.
tf.config.set_visible_devices([], 'GPU')
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ── 설정 ──────────────────────────────────────────────
YOLO_SEG_WEIGHT = 'yolo11n-bed-seg.pt'  # hospital bed fine-tuned (class 0)
YOLO_SEG_CLASS = 0
YOLO_POSE_WEIGHT = 'yolo11m-pose.pt'
YOLO_DEVICE = os.environ.get('POSE_YOLO_DEVICE', '0')
SEG_EVERY_N = max(1, int(os.environ.get('POSE_SEG_EVERY', '3')))
RTSP_URL = os.environ.get('POSE_RTSP_URL', 'rtsp://192.168.0.161:8554/stream')
FRAME_WIDTH = int(os.environ.get('POSE_FRAME_WIDTH', '640'))  # extracted_frames / RTSP 기준 640×360
VIEWER_SCALE = float(os.environ.get('POSE_VIEWER_SCALE', '1.5'))  # /viewer CSS scale vs FRAME_WIDTH
HUD_FONT = float(os.environ.get('POSE_HUD_FONT', '0.38'))
HUD_FONT_SM = float(os.environ.get('POSE_HUD_FONT_SM', '0.32'))
YOLO_PLOT_LINE_WIDTH = int(os.environ.get('POSE_YOLO_LINE_WIDTH', '1'))
YOLO_PLOT_KPT_RADIUS = int(os.environ.get('POSE_YOLO_KPT_RADIUS', '2'))
YOLO_PLOT_LABELS = os.environ.get('POSE_YOLO_LABELS', '0') == '1'
YOLO_PLOT_BOXES = os.environ.get('POSE_YOLO_BOXES', '0') == '1'
SHOW_BED_MASK = os.environ.get('POSE_SHOW_BED_MASK', '1') != '0'
SHOW_BED_ROI = os.environ.get('POSE_SHOW_BED_ROI', '0') == '1'
SHOW_BED_ZONES = os.environ.get('POSE_SHOW_BED_ZONES', '0') == '1'
SHOW_BED_BBOX = os.environ.get('POSE_SHOW_BED_BBOX', '0') == '1'
SHOW_RAIL_ROI = os.environ.get('POSE_SHOW_RAIL_ROI', '0') == '1'
RTSP_HOST = urlparse(RTSP_URL).hostname or ''
BED_ROI_PATH = os.environ.get('POSE_BED_ROI', '/home/dmc/AI/DMC_POSE/bed_roi/bed_roi.json')
USE_BED_ROI = os.environ.get('POSE_USE_BED_ROI', '1') != '0'
USE_RAIL = os.environ.get('POSE_USE_RAIL', '1') != '0'
RAIL_CONFIG_PATH = os.environ.get(
    'POSE_RAIL_CONFIG',
    str(Path(__file__).resolve().parent / 'rail' / 'rail_config.json'),
)
POSE_KERAS_MODEL = 'my_model_six_check.keras'
CLASS_NAMES = [
    "정면_누움",
    "엎드림_등",
    "옆누움_가까움",
    "옆누움_멀음",
    "앉음_중앙",
    "앉음_가장자리",
]
CLASS_DISPLAY_NAMES = [
    "front_lying",
    "prone_back",
    "side_near",
    "side_far",
    "sitting_center",
    "sitting_edge",
]

# ── 공유 상태 ─────────────────────────────────────────
state_lock = Lock()
shared_state = {
    "in_bed": "NO",
    "pose": "None",
    "pose_conf": 0.0,
    "rail_left_up": False,
    "rail_left_score": 0.0,
    "rail_right_up": False,
    "rail_right_score": 0.0,
    "timestamp": None,
    "latency_ms": 0.0,
    "frame_age_ms": 0.0,
    "pipeline_fps": 0.0,
    "is_running": False,
    "frame_jpeg": None,
    "in_bed_method": "none",
    "person_detected": False,
    "seg_attachment": "none",
    "kpt_on_seg_ratio": 0.0,
    "edge_zone": None,
    "limb_overflow_max": 0.0,
    "risk_level": "SAFE",
    "center_speed": None,
    "bed_event": None,
    "last_bed_event": None,
    "bed_source": "none",
    "skeleton_kpts": 0,
    "skeleton_core_kpts": 0,
    "zone_quality": "none",
    "bed_source": "none",
    "fall_score": 0.0,
    "fall_level": "SAFE",
    "fall_status": "NO_PERSON",
    "rail_risk": 0.0,
    "zone_risk": 0.0,
    "pose_risk": 0.0,
}

# ── 응답 모델 ─────────────────────────────────────────
class PoseStatus(BaseModel):
    in_bed: str
    pose: str
    pose_conf: float
    rail_left_up: bool
    rail_left_score: float
    rail_right_up: bool
    rail_right_score: float
    timestamp: datetime | None
    ip_address: str
    latency_ms: float
    frame_age_ms: float
    pipeline_fps: float
    in_bed_method: str = "none"
    person_detected: bool = False
    seg_attachment: str = "none"
    kpt_on_seg_ratio: float = 0.0
    edge_zone: str | None = None
    limb_overflow_max: float = 0.0
    risk_level: str = "SAFE"
    center_speed: float | None = None
    bed_event: str | None = None
    last_bed_event: str | None = None
    zone_quality: str = "none"
    bed_source: str = "none"
    fall_score: float = 0.0
    fall_level: str = "SAFE"
    fall_status: str = "NO_PERSON"
    rail_risk: float = 0.0
    zone_risk: float = 0.0
    pose_risk: float = 0.0

class HealthStatus(BaseModel):
    server: str
    analysis_running: bool

# ── 침대 seg / bbox ─────────────────────────────────────
MASK_THRESH = float(os.environ.get('POSE_BED_MASK_THRESH', '0.35'))
BBOX_PAD_RATIO = float(os.environ.get('POSE_BED_BBOX_PAD', '0.02'))
BED_SEG_CONF = float(os.environ.get('POSE_BED_SEG_CONF', '0.01'))
MIN_BED_AREA = float(os.environ.get('POSE_BED_MIN_AREA', '0.08'))
MAX_BED_AREA = float(os.environ.get('POSE_BED_MAX_AREA', '0.55'))


def _mask_to_frame(mask, h: int, w: int) -> np.ndarray:
    m_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.asarray(mask)
    if m_np.ndim == 3:
        m_np = m_np[0]
    if m_np.shape[0] != h or m_np.shape[1] != w:
        m_np = cv2.resize(m_np.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return m_np


def _refine_binary_mask(m_np: np.ndarray) -> np.ndarray | None:
    binary = (m_np > MASK_THRESH).astype(np.uint8)
    if not binary.any():
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    refined = (labels == largest).astype(np.uint8)
    area_ratio = refined.sum() / refined.size
    if area_ratio < MIN_BED_AREA or area_ratio > MAX_BED_AREA:
        return None
    return refined


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    cols = np.any(mask > 0, axis=0)
    rows = np.any(mask > 0, axis=1)
    if not cols.any() or not rows.any():
        return None
    x_min = int(np.argmax(cols))
    x_max = int(len(cols) - np.argmax(cols[::-1]) - 1)
    y_min = int(np.argmax(rows))
    y_max = int(len(rows) - np.argmax(rows[::-1]) - 1)
    return x_min, y_min, x_max, y_max


def _bbox_from_boxes(seg_result, h: int, w: int) -> tuple[int, int, int, int] | None:
    if seg_result.boxes is None or len(seg_result.boxes) == 0:
        return None
    best_idx = int(seg_result.boxes.conf.argmax())
    x1, y1, x2, y2 = seg_result.boxes.xyxy[best_idx].cpu().numpy()
    return int(x1), int(y1), int(x2), int(y2)


def _pad_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    pad_x = int((x_max - x_min) * BBOX_PAD_RATIO)
    pad_y = int((y_max - y_min) * BBOX_PAD_RATIO)
    return (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(w - 1, x_max + pad_x),
        min(h - 1, y_max + pad_y),
    )


def extract_bed_detection(seg_result, h: int, w: int) -> dict:
    """Best-confidence detection only — mask 우선, 실패 시 box bbox 폴백."""
    refined_mask = None
    bbox = None
    source = 'none'

    if seg_result.boxes is None or len(seg_result.boxes) == 0:
        return {'mask': None, 'bbox': None, 'source': 'none'}

    best_idx = int(seg_result.boxes.conf.argmax())

    if seg_result.masks is not None and len(seg_result.masks) > best_idx:
        m_np = _mask_to_frame(seg_result.masks.data[best_idx], h, w)
        refined_mask = _refine_binary_mask(m_np)
        if refined_mask is not None:
            bbox = _bbox_from_mask(refined_mask)
            if bbox is not None:
                source = 'mask'

    if bbox is None:
        x1, y1, x2, y2 = seg_result.boxes.xyxy[best_idx].cpu().numpy()
        bbox = _pad_bbox((int(x1), int(y1), int(x2), int(y2)), h, w)
        source = 'box'

    if bbox is not None:
        bbox = _pad_bbox(bbox, h, w)

    return {
        'mask': refined_mask,
        'bbox': bbox,
        'source': source,
    }


def is_person_in_bed(person_bbox, bed: dict, h: int, w: int) -> tuple[bool, str]:
    x1, y1, x2, y2 = person_bbox[:4]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    mask = bed.get('mask')
    if mask is not None and 0 <= cy < h and 0 <= cx < w and mask[cy, cx] > 0:
        return True, 'mask'

    bbox = bed.get('bbox')
    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            return True, 'bbox'

    return False, 'none'


def draw_bed_zones(frame: np.ndarray, bed_bbox) -> np.ndarray:
    if bed_bbox is None:
        return frame
    x_min, y_min, x_max, y_max = bed_bbox
    bed_w = x_max - x_min
    x1_3 = x_min + bed_w // 3
    x2_3 = x_min + 2 * bed_w // 3
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (220, 220, 220), 2)
    cv2.line(frame, (x1_3, y_min), (x1_3, y_max), (220, 220, 220), 1)
    cv2.line(frame, (x2_3, y_min), (x2_3, y_max), (220, 220, 220), 1)
    cy = (y_min + y_max) // 2
    for label, x in (('L', x_min + 4), ('C', x1_3 + 4), ('R', x2_3 + 4)):
        cv2.putText(
            frame, label, (x, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


def draw_compact_hud(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
) -> np.ndarray:
    """Small semi-transparent panel at bottom-left — avoids covering the bed view."""
    if not lines:
        return frame
    h, w = frame.shape[:2]
    pad = 4
    line_h = max(12, int(h * 0.032))
    box_h = pad * 2 + line_h * len(lines)
    box_w = min(w - pad * 2, int(w * 0.78))
    y0 = h - box_h - pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, y0), (pad + box_w, h - pad), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    y = y0 + pad + line_h - 3
    for text, color in lines:
        cv2.putText(
            out, text, (pad + 6, y), cv2.FONT_HERSHEY_SIMPLEX, HUD_FONT_SM, color, 1, cv2.LINE_AA,
        )
        y += line_h
    return out


def draw_mask_contours(mask: np.ndarray, frame: np.ndarray, downscale:int=2, color=(60,200,120), alpha:float=0.22) -> np.ndarray:
    try:
        fh, fw = frame.shape[:2]
        m = mask
        if m.max() > 1 and m.max() <= 255:
            _, m_bin = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
        else:
            m_bin = (m > 0).astype('uint8') * 255
        if downscale > 1:
            small = cv2.resize(m_bin, (max(1, fw//downscale), max(1, fh//downscale)), interpolation=cv2.INTER_NEAREST)
        else:
            small = m_bin
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame
        scaled_contours = []
        sx = fw / small.shape[1]
        sy = fh / small.shape[0]
        for c in contours:
            c = c.reshape(-1,2).astype('float32')
            c[:,0] = np.clip(np.round(c[:,0] * sx).astype(int), 0, fw-1)
            c[:,1] = np.clip(np.round(c[:,1] * sy).astype(int), 0, fh-1)
            scaled_contours.append(c.reshape(-1,1,2).astype('int32'))
        overlay = frame.copy()
        cv2.drawContours(overlay, scaled_contours, -1, color, thickness=cv2.FILLED)
        out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return out
    except Exception as e:
        logging.warning(f"draw_mask_contours error: {e}")
        return frame


def draw_bed_overlay(frame: np.ndarray, bed: dict, roi_bbox=None) -> np.ndarray:
    out = frame.copy()
    if SHOW_BED_ROI and roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    mask = bed.get('mask')
    if SHOW_BED_MASK and mask is not None:
        # optimized contour overlay when enabled
        if os.environ.get('POSE_SHOW_BED_MASK_OPTIMIZED', '1') == '1':
            out = draw_mask_contours(mask, out, downscale=2, color=(60,200,120), alpha=0.22)
        else:
            color = np.zeros_like(out)
            color[mask > 0] = (60, 200, 120)
            out = cv2.addWeighted(out, 1.0, color, 0.22, 0)
    elif SHOW_BED_BBOX and bed.get('bbox') is not None:
        x1, y1, x2, y2 = bed['bbox']
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 200, 120), 1)
    if SHOW_BED_ZONES:
        out = draw_bed_zones(out, bed.get('bbox'))
    return out


class RTSPReader:
    """Always keep the latest RTSP frame; drop buffered older frames."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.lock = Lock()
        self.frame: np.ndarray | None = None
        self.captured_at = 0.0
        self.running = True
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        cap = self._open()
        while self.running:
            # Flush decode buffer — grab stale frames without decoding.
            for _ in range(4):
                if not cap.grab():
                    break
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                time.sleep(2)
                cap = self._open()
                continue
            with self.lock:
                self.frame = frame
                self.captured_at = time.perf_counter()

    def read_latest(self) -> tuple[bool, np.ndarray | None, float]:
        with self.lock:
            if self.frame is None:
                return False, None, 0.0
            return True, self.frame.copy(), self.captured_at

    def stop(self) -> None:
        self.running = False


# ── 분석 루프 (백그라운드 스레드) ─────────────────────
def analysis_loop():
    logging.info("-" * 50)
    logging.info(f"[INFO] YOLO GPU={YOLO_DEVICE} | seg every {SEG_EVERY_N} frames")
    logging.info(f"[INFO] RTSP {RTSP_URL}")
    if USE_BED_ROI:
        logging.info(f"[INFO] bed ROI {BED_ROI_PATH}")
    if USE_RAIL:
        logging.info(f"[INFO] rail detect {RAIL_CONFIG_PATH}")
    logging.info("-" * 50)

    if not os.path.exists(POSE_KERAS_MODEL):
        logging.error(f"❌ 모델 파일 없음: {POSE_KERAS_MODEL}")
        return

    try:
        yolo_seg_model = YOLO(YOLO_SEG_WEIGHT)
        yolo_pose_model = YOLO(YOLO_POSE_WEIGHT)
        keras_clf = keras.models.load_model(POSE_KERAS_MODEL)
    except Exception as e:
        logging.error(f"❌ 로딩 에러: {e}")
        return

    with state_lock:
        shared_state["is_running"] = True

    stream = RTSPReader(RTSP_URL)
    frame_idx = 0
    cached_bed: dict | None = None
    ema_fps = 0.0
    bed_source = 'none'
    in_bed_method = 'none'
    rail_cfg = load_rail_config(Path(RAIL_CONFIG_PATH)) if USE_RAIL else None
    preset = load_preset()
    motion_state = MotionState()
    event_tracker = LiveEventTracker(preset)
    scoring_cfg = preset.get("scoring") or {}
    fall_scorer = FallScorer(scoring_cfg) if scoring_cfg.get("enabled") else None
    analysis_t0: float | None = None
    last_kxy: np.ndarray | None = None
    last_kconf: np.ndarray | None = None

    while True:
        loop_start = time.perf_counter()
        ret, frame, captured_at = stream.read_latest()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frame = imutils.resize(frame, width=FRAME_WIDTH)
        fh, fw = frame.shape[:2]
        frame_idx += 1

        if frame_idx % SEG_EVERY_N == 1 or cached_bed is None:
            seg_res = yolo_seg_model.predict(
                frame,
                classes=[YOLO_SEG_CLASS],
                conf=BED_SEG_CONF,
                device=YOLO_DEVICE,
                verbose=False,
            )
            fresh_bed = extract_bed_detection(seg_res[0], fh, fw)
            if fresh_bed.get('bbox') is not None or fresh_bed.get('mask') is not None:
                cached_bed = fresh_bed
        bed = cached_bed or {'mask': None, 'bbox': None, 'source': 'none'}
        roi_bbox = load_bed_roi(Path(BED_ROI_PATH), fw, fh) if USE_BED_ROI else None
        if roi_bbox is not None:
            bed = apply_bed_roi(bed, roi_bbox, fh, fw)
        if preset.get("bed_zone"):
            bed = build_approx_bed_zone(bed, roi_bbox, fh, fw, preset)
        bed_source = bed.get("source", "none")
        zone_quality = bed.get("zone_quality", "none")

        pose_res = yolo_pose_model.predict(
            frame, conf=0.5, device=YOLO_DEVICE, verbose=False
        )

        in_bed = "NO"
        pose = "None"
        pose_display = "None"
        pose_conf = 0.0
        in_bed_method = "none"
        person_detected = False
        seg_attachment = "none"
        kpt_on_seg_ratio = 0.0
        skeleton_kpts = 0
        skeleton_core_kpts = 0
        skeleton_kpts_need = 5
        skeleton_core_need = 1
        yolo_pose_raw = False
        person_bbox = None
        edge_zone = None
        limb_overflow = 0.0
        risk_level = "SAFE"
        center_speed = None
        bed_event = None
        last_bed_event = event_tracker.last_event
        fall_score = 0.0
        fall_level = "SAFE"
        fall_status = "NO_PERSON"
        rail_risk = 0.0
        zone_risk = 0.0
        pose_risk = 0.0
        feat: dict = {}
        last_kxy = None
        last_kconf = None
        display_frame = draw_bed_overlay(frame, bed, roi_bbox)

        if analysis_t0 is None:
            analysis_t0 = loop_start
        t_sec = loop_start - analysis_t0

        if len(pose_res) > 0 and len(pose_res[0].keypoints) > 0:
            yolo_pose_raw = True
            kp = pose_res[0].keypoints[0]
            kxy = kp.xy.cpu().numpy().reshape(-1, 2)
            if kp.conf is not None:
                kconf = kp.conf.cpu().numpy().flatten()
            else:
                kconf = np.ones(len(kxy), dtype=np.float32)

            last_kxy = kxy
            last_kconf = kconf

            feat = enrich_from_keypoints(kxy, kconf, bed, motion_state, t_sec, preset)
            person_detected = bool(feat["person_detected"])
            seg_attachment = feat["seg_attachment"]
            kpt_on_seg_ratio = float(feat["kpt_on_seg_ratio"])
            skeleton_kpts = int(feat.get("skeleton_kpts", 0))
            skeleton_core_kpts = int(feat.get("skeleton_core_kpts", 0))
            skeleton_kpts_need = int(feat.get("skeleton_kpts_need", 5))
            skeleton_core_need = int(feat.get("skeleton_core_need", 1))
            bed_source = feat.get("bed_source", bed_source)
            zone_quality = feat.get("zone_quality", zone_quality)
            in_bed = "YES" if feat["in_bed"] else "NO"
            in_bed_method = feat["in_bed_method"]
            edge_zone = feat["edge_zone"]
            limb_overflow = feat["limb_overflow_max"]
            risk_level = feat["risk_level"]
            center_speed = feat["center_speed"]
            event_tracker.update(t_sec, feat)
            bed_event = event_tracker.bed_event
            last_bed_event = event_tracker.last_event

            if person_detected:
                display_frame = pose_res[0].plot(
                    img=display_frame,
                    line_width=YOLO_PLOT_LINE_WIDTH,
                    font_size=10,
                    kpt_radius=YOLO_PLOT_KPT_RADIUS,
                    labels=YOLO_PLOT_LABELS,
                    boxes=YOLO_PLOT_BOXES,
                    conf=False,
                    masks=False,
                )
                person_bbox = pose_res[0].boxes.xyxy[0].cpu().numpy()

                kpts_data = kxy.flatten()
                if len(kpts_data) == 34:
                    kpts_input = kpts_data.reshape(1, -1).astype('float32')
                    pred = keras_clf.predict(kpts_input, verbose=0)
                    pred_idx = int(np.argmax(pred))
                    pose = CLASS_NAMES[pred_idx]
                    pose_display = CLASS_DISPLAY_NAMES[pred_idx]
                    pose_conf = float(pred[0][pred_idx])

        rail_left_up = False
        rail_left_score = 0.0
        rail_right_up = False
        rail_right_score = 0.0
        rail_left_method = "off"
        rail_right_method = "off"
        if USE_RAIL and rail_cfg is not None:
            with state_lock:
                prev_left = shared_state["rail_left_up"]
                prev_right = shared_state["rail_right_up"]
            rail_result = detect_both_rails(
                frame,
                rail_cfg,
                person_xyxy=person_bbox,
                prev_left=prev_left,
                prev_right=prev_right,
                pose_label=pose if pose != "None" else None,
            )
            rail_left_up = bool(rail_result["rail_left_up"])
            rail_left_score = float(rail_result["rail_left_score"])
            rail_right_up = bool(rail_result["rail_right_up"])
            rail_right_score = float(rail_result["rail_right_score"])
            rail_left_method = str(rail_result["rail_left_method"])
            rail_right_method = str(rail_result["rail_right_method"])
            if SHOW_RAIL_ROI:
                display_frame = draw_rail_rois(
                    display_frame, rail_cfg, rail_left_up, rail_right_up
                )

        if fall_scorer is not None and last_kxy is not None and last_kconf is not None:
            if not feat:
                feat = {
                    "person_detected": person_detected,
                    "seg_attachment": seg_attachment,
                    "in_bed": in_bed == "YES",
                    "edge_zone": edge_zone,
                }
            apply_fall_scoring(
                feat,
                last_kxy,
                last_kconf,
                bed.get("bbox"),
                preset,
                fall_scorer,
                pose_ko=pose if pose != "None" else None,
                rail_left_up=rail_left_up if USE_RAIL else None,
                rail_right_up=rail_right_up if USE_RAIL else None,
            )
            fall_score = float(feat.get("fall_score", 0.0))
            fall_level = str(feat.get("fall_level", "SAFE"))
            fall_status = str(feat.get("fall_status", "NO_PERSON"))
            rail_risk = float(feat.get("rail_risk", 0.0))
            zone_risk = float(feat.get("zone_risk", 0.0))
            pose_risk = float(feat.get("pose_risk", 0.0))

        loop_end = time.perf_counter()
        latency_ms = (loop_end - loop_start) * 1000.0
        frame_age_ms = (loop_end - captured_at) * 1000.0 if captured_at else latency_ms
        instant_fps = 1.0 / max(loop_end - loop_start, 1e-6)
        ema_fps = instant_fps if ema_fps == 0.0 else (0.85 * ema_fps + 0.15 * instant_fps)

        if person_detected:
            person_lbl = "YES"
            attach_lbl = f"{seg_attachment} ({kpt_on_seg_ratio:.0%})"
        elif yolo_pose_raw:
            person_lbl = (
                f"weak {skeleton_kpts}/{skeleton_kpts_need}"
                f" core{skeleton_core_kpts}/{skeleton_core_need}"
            )
            attach_lbl = "n/a"
        else:
            person_lbl = "NO"
            attach_lbl = "n/a"

        hud_lines: list[tuple[str, tuple[int, int, int]]] = [
            (
                f"Zone:{zone_quality}  Person:{person_lbl}  Attach:{attach_lbl}  "
                f"Pose:{pose_display} ({pose_conf:.2f})",
                (0, 255, 255),
            ),
        ]
        ov_pct = limb_overflow * 100.0
        spd = f"{center_speed:.0f}" if center_speed is not None else "-"
        edge_lbl = edge_zone or "-"
        risk_color = (0, 255, 255) if risk_level == "SAFE" else (0, 165, 255)
        hud_lines.append(
            (f"edge:{edge_lbl}  overflow:{ov_pct:.0f}%  risk:{risk_level}  spd:{spd}", risk_color),
        )
        if fall_scorer is not None:
            score_color = (0, 255, 255) if fall_level == "SAFE" else (0, 140, 255)
            hud_lines.append(
                (f"score:{fall_score:.0f} {fall_level} [{fall_status}]", score_color),
            )
        if bed_event:
            hud_lines.append((f"** EVENT: {bed_event} **", (0, 80, 255)))
        elif last_bed_event:
            hud_lines.append((f"last event: {last_bed_event}", (180, 180, 180)))
        if USE_RAIL:
            rail_r_label = "UP" if rail_right_up else "DOWN"
            rail_l_label = "UP" if rail_left_up else "DOWN"
            hud_lines.append(
                (
                    f"Rail R:{rail_r_label} L:{rail_l_label}  "
                    f"{latency_ms:.0f}ms {ema_fps:.0f}fps",
                    (200, 200, 200),
                ),
            )
        else:
            hud_lines.append(
                (f"{latency_ms:.0f}ms | {ema_fps:.0f}fps", (200, 200, 200)),
            )
        display_frame = draw_compact_hud(display_frame, hud_lines)
        ok, buf = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_jpeg = buf.tobytes() if ok else None

        now = datetime.now()
        with state_lock:
            shared_state["in_bed"] = in_bed
            shared_state["pose"] = pose
            shared_state["pose_conf"] = pose_conf
            shared_state["in_bed_method"] = in_bed_method
            shared_state["person_detected"] = person_detected
            shared_state["seg_attachment"] = seg_attachment
            shared_state["kpt_on_seg_ratio"] = round(kpt_on_seg_ratio, 4)
            shared_state["bed_source"] = bed_source
            shared_state["zone_quality"] = zone_quality
            shared_state["skeleton_kpts"] = skeleton_kpts
            shared_state["skeleton_core_kpts"] = skeleton_core_kpts
            shared_state["edge_zone"] = edge_zone
            shared_state["limb_overflow_max"] = round(limb_overflow, 4)
            shared_state["risk_level"] = risk_level
            shared_state["center_speed"] = (
                round(center_speed, 2) if center_speed is not None else None
            )
            shared_state["bed_event"] = bed_event
            shared_state["last_bed_event"] = last_bed_event
            shared_state["rail_left_up"] = rail_left_up
            shared_state["rail_left_score"] = round(rail_left_score, 4)
            shared_state["rail_right_up"] = rail_right_up
            shared_state["rail_right_score"] = round(rail_right_score, 4)
            shared_state["fall_score"] = round(fall_score, 2)
            shared_state["fall_level"] = fall_level
            shared_state["fall_status"] = fall_status
            shared_state["rail_risk"] = round(rail_risk, 4)
            shared_state["zone_risk"] = round(zone_risk, 4)
            shared_state["pose_risk"] = round(pose_risk, 4)
            shared_state["timestamp"] = now
            shared_state["latency_ms"] = round(latency_ms, 1)
            shared_state["frame_age_ms"] = round(frame_age_ms, 1)
            shared_state["pipeline_fps"] = round(ema_fps, 1)
            shared_state["frame_jpeg"] = frame_jpeg

        # logging.info(f"In Bed: {in_bed:3} | Pose: {pose} | age {frame_age_ms:.0f}ms")

# ── FastAPI 앱 ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = Thread(target=analysis_loop, daemon=True)
    thread.start()
    yield

app = FastAPI(
    title="Pose Monitoring API",
    description="침대 위 포즈 모니터링 실시간 상태 조회",
    lifespan=lifespan,
)

@app.get("/status", response_model=PoseStatus)
def get_status():
    """최신 포즈 분석 결과를 반환합니다."""
    with state_lock:
        return PoseStatus(
            in_bed=shared_state["in_bed"],
            pose=shared_state["pose"],
            pose_conf=shared_state["pose_conf"],
            rail_left_up=shared_state["rail_left_up"],
            rail_left_score=shared_state["rail_left_score"],
            rail_right_up=shared_state["rail_right_up"],
            rail_right_score=shared_state["rail_right_score"],
            timestamp=shared_state["timestamp"],
            ip_address=RTSP_HOST,
            latency_ms=shared_state["latency_ms"],
            frame_age_ms=shared_state["frame_age_ms"],
            pipeline_fps=shared_state["pipeline_fps"],
            in_bed_method=shared_state.get("in_bed_method", "none"),
            edge_zone=shared_state.get("edge_zone"),
            limb_overflow_max=shared_state.get("limb_overflow_max", 0.0),
            risk_level=shared_state.get("risk_level", "SAFE"),
            center_speed=shared_state.get("center_speed"),
            bed_event=shared_state.get("bed_event"),
            last_bed_event=shared_state.get("last_bed_event"),
            person_detected=shared_state.get("person_detected", False),
            seg_attachment=shared_state.get("seg_attachment", "none"),
            kpt_on_seg_ratio=shared_state.get("kpt_on_seg_ratio", 0.0),
            zone_quality=shared_state.get("zone_quality", "none"),
            bed_source=shared_state.get("bed_source", "none"),
            fall_score=shared_state.get("fall_score", 0.0),
            fall_level=shared_state.get("fall_level", "SAFE"),
            fall_status=shared_state.get("fall_status", "NO_PERSON"),
            rail_risk=shared_state.get("rail_risk", 0.0),
            zone_risk=shared_state.get("zone_risk", 0.0),
            pose_risk=shared_state.get("pose_risk", 0.0),
        )

@app.get("/health", response_model=HealthStatus)
def get_health():
    """서버 및 분석 스레드 상태를 확인합니다."""
    with state_lock:
        running = shared_state["is_running"]
    return HealthStatus(server="ok", analysis_running=running)


def mjpeg_generator():
    while True:
        with state_lock:
            frame = shared_state["frame_jpeg"]
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.01)


@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/viewer", response_class=HTMLResponse)
def viewer_page():
    viewer_w = int(FRAME_WIDTH * VIEWER_SCALE)
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Pose Viewer (Test)</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 8px; background: #111; color: #eee; }}
          h3 {{ margin: 0 0 8px 0; font-size: 14px; font-weight: normal; color: #888; }}
          img {{ width: {viewer_w}px; max-width: 100%; height: auto; border: 1px solid #444; display: block; }}
        </style>
      </head>
      <body>
        <h3>Pose Viewer</h3>
        <img src="/video" alt="Live stream" />
      </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn

    # LAN에서 http://<이머신IP>:8000/viewer 접속 시 모든 인터페이스에 바인딩 필요
    _host = os.environ.get("POSE_API_HOST", "0.0.0.0")
    _port = int(os.environ.get("POSE_API_PORT", "8000"))
    uvicorn.run(app, host=_host, port=_port)
