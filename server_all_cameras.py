"""
4 Camera Parallel GPU Inference Server (Unified Philosophy)
RTSP Multi-Camera Pipeline + Original Rule-based Logic
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|hwaccel;cuda|flags2;+showall|analyzeduration;500000|probesize;1000000"

import cv2
import numpy as np
import tensorflow as tf
import keras
import logging
import time
import json
from pathlib import Path
from ultralytics import YOLO, SAM
from threading import Lock, Thread
from datetime import datetime
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

# [IMPORT ORIGINAL PHILOSOPHY]
from bed_monitor.config import load_preset
from bed_monitor.features import MotionState
from bed_monitor.temporal import LiveEventTracker
from bed_monitor.scoring import FallScorer
from bed_monitor.live import enrich_from_keypoints
from bed_monitor.bed_zone import build_approx_bed_zone

# [IMPORT NEW MODULES]
from frame_buffer import FrameBuffer
from analysis_state_machine import CameraAnalysisStateMachine, AnalysisState as AnalysisStateEnum
from live_temporal import TemporalModelService, TemporalShadowRunner
from latest_frame_capture import LatestFrameCapture
from auto_bed_roi import AutoBedROIManager
from motion_watcher import MotionWatcher
from edge_signal_client import EdgeSignalClient
from edge_managed_policy import edge_runtime_policy, parse_camera_ids
from person_tracker import MultiPersonTracker, PersonDetection, keypoints_bbox
from pre_event_replay import ReplayPoseFrame, replay_temporal_context
from hybrid_fusion import HybridFusion, FusionInput
from shadow_feature_recorder import ShadowFeatureRecorder
from temporal_session_recorder import (
    TemporalEventSessionRecorder,
    derive_temporal_session_triggers,
)
from pose_candidate_filter import accept_pose_candidate, select_tracking_bbox
from async_replay_worker import AsyncReplayWorker
from runtime_health import evaluate_fleet, render_prometheus
from inference_scheduler import (
    LatestInferenceScheduler,
    P0_VERIFY,
    P1_BURST,
    P2_OCCUPIED,
    P3_EMPTY_PROBE,
    P4_BED_SEG,
)
from spatial_geometry import (
    orient_bed_detection,
    select_refined_bed_candidate,
    skeleton_bed_coverage,
)

# Keras only on CPU; YOLO uses PyTorch GPU
tf.config.set_visible_devices([], 'GPU')
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ── 6 카메라 설정 ──────────────────────────────────────────────
CAMERA_CONFIGS = {
    'bed_161': {'name': 'Bed 161', 'rtsp_url': 'rtsp://192.168.0.161:8554/stream'},
    'bed_162': {'name': 'Bed 162', 'rtsp_url': 'rtsp://192.168.0.162:8554/stream'},
    'bed_174': {'name': 'Bed 174', 'rtsp_url': 'rtsp://192.168.0.174:8554/stream'},
    'bed_175': {'name': 'Bed 175', 'rtsp_url': 'rtsp://192.168.0.175:8554/stream'},
    'bed_178': {'name': 'Bed 178', 'rtsp_url': 'rtsp://192.168.0.178:8554/stream'},
    'bed_179': {'name': 'Bed 179', 'rtsp_url': 'rtsp://192.168.0.179:8554/stream'},
}

# ── 글로벌 설정 ──────────────────────────────────────────────
YOLO_SEG_WEIGHT = os.environ.get(
    'POSE_YOLO_SEG_WEIGHT',
    str(Path(__file__).resolve().parent / 'bed_seg/runs/bed_seg/weights/best.pt'),
)
YOLO_SEG_CLASS = 0
YOLO_POSE_WEIGHT = 'yolo11m-pose.pt'
YOLO_DEVICE = os.environ.get('POSE_YOLO_DEVICE', '0')
SEG_EVERY_N = max(1, int(os.environ.get('POSE_SEG_EVERY', '3')))
FRAME_WIDTH = int(os.environ.get('POSE_FRAME_WIDTH', '640'))
ANALYSIS_ROTATION = int(os.environ.get('POSE_ANALYSIS_ROTATION', '90')) % 360
if ANALYSIS_ROTATION not in (0, 90, 180, 270):
    raise ValueError('POSE_ANALYSIS_ROTATION must be one of 0, 90, 180, 270')
BED_SEG_CONF = float(os.environ.get('POSE_BED_SEG_CONF', '0.1'))
BED_REFINER_ENABLED = os.environ.get('POSE_BED_REFINER', '1') == '1'
BED_REFINER_WEIGHT = os.environ.get(
    'POSE_BED_REFINER_WEIGHT',
    str(Path(__file__).resolve().parent / 'mobile_sam.pt'),
)
BED_REFINER_DEVICE = os.environ.get('POSE_BED_REFINER_DEVICE', 'cpu')
BED_REFINER_MIN_AREA_RATIO = float(os.environ.get(
    'POSE_BED_REFINER_MIN_AREA_RATIO', '0.04'
))
BED_REFINER_MAX_AREA_RATIO = float(os.environ.get(
    'POSE_BED_REFINER_MAX_AREA_RATIO', '0.65'
))
BED_REFINER_MIN_EXTENT_RATIO = float(os.environ.get(
    'POSE_BED_REFINER_MIN_EXTENT_RATIO', '0.40'
))
PERSON_POSE_CONF = float(os.environ.get('POSE_PERSON_CONF', '0.03'))
REPLAY_PERSON_POSE_CONF = float(os.environ.get(
    'POSE_REPLAY_PERSON_CONF', '0.03'
))
POSE_STRONG_BOX_CONF = float(os.environ.get('POSE_STRONG_BOX_CONF', '0.5'))
POSE_STRONG_MIN_AREA_RATIO = float(os.environ.get(
    'POSE_STRONG_MIN_AREA_RATIO', '0.016'
))
POSE_WEAK_BOX_CONF = float(os.environ.get('POSE_WEAK_BOX_CONF', '0.05'))
POSE_WEAK_MIN_VISIBLE = int(os.environ.get('POSE_WEAK_MIN_VISIBLE', '8'))
POSE_WEAK_MIN_KP_MEAN = float(os.environ.get('POSE_WEAK_MIN_KP_MEAN', '0.25'))
POSE_WEAK_MIN_AREA_RATIO = float(os.environ.get(
    'POSE_WEAK_MIN_AREA_RATIO', '0.025'
))
PARALLEL_WORKERS = int(os.environ.get('POSE_PARALLEL_WORKERS', '3'))
INFERENCE_URGENT_QUOTA = int(os.environ.get('POSE_INFERENCE_URGENT_QUOTA', '4'))
POSE_KERAS_MODEL = 'my_model_six_check.keras'
# No per-camera manual ROI is used in production. Each camera builds and
# validates its own bed mask from repeated segmentation observations.
# optimized mask overlay: draw contours on downscaled mask to reduce CPU, set 0/1
SHOW_BED_MASK_OPTIMIZED = os.environ.get('POSE_SHOW_BED_MASK_OPTIMIZED', '1') == '1'
BODY_IN_BED_SAFE_THRESHOLD = float(os.environ.get('POSE_BODY_IN_BED_SAFE_THRESHOLD', '0.80'))
PROJECT_ROOT = Path(__file__).resolve().parent
AUTO_BED_CACHE_DIR = Path(os.environ.get(
    'POSE_AUTO_BED_CACHE_DIR', PROJECT_ROOT / 'bed_roi/auto_cache'
))
AUTO_BED_CANDIDATE_WINDOW = int(os.environ.get('POSE_AUTO_BED_WINDOW', '5'))
AUTO_BED_MIN_DETECTIONS = int(os.environ.get('POSE_AUTO_BED_MIN_DETECTIONS', '3'))
AUTO_BED_CONSENSUS_IOU = float(os.environ.get('POSE_AUTO_BED_CONSENSUS_IOU', '0.75'))
AUTO_BED_REFRESH_SEC = float(os.environ.get('POSE_AUTO_BED_REFRESH_SEC', '300'))
AUTO_BED_SCENE_CHANGE_RATIO = float(os.environ.get('POSE_AUTO_BED_SCENE_CHANGE_RATIO', '0.75'))
AUTO_BED_SCENE_CHANGE_PERSISTENCE = int(os.environ.get(
    'POSE_AUTO_BED_SCENE_CHANGE_PERSISTENCE', '3'
))
TCN_SHADOW_ENABLED = os.environ.get('POSE_TCN_SHADOW', '1') == '1'
TCN_MODEL_PATH = Path(os.environ.get(
    'POSE_TCN_MODEL', PROJECT_ROOT / 'runs/temporal_tcn/gmdcsa24_tcn/model.pt'
))
TCN_REPORT_PATH = Path(os.environ.get(
    'POSE_TCN_REPORT', PROJECT_ROOT / 'runs/temporal_tcn/gmdcsa24_tcn/report.json'
))
TCN_DEVICE = os.environ.get('POSE_TCN_DEVICE', 'cpu')
TCN_THRESHOLD = os.environ.get('POSE_TCN_THRESHOLD')
TCN_ALLOW_NON_PROMOTION = os.environ.get(
    'POSE_TCN_ALLOW_NON_PROMOTION', '0'
) == '1'
TCN_FUSION_ENABLED = os.environ.get('POSE_TCN_FUSION_ENABLED', '1') == '1'
CENTRAL_POSE_ALWAYS_ON = os.environ.get(
    'POSE_CENTRAL_ALWAYS_ON', '0'
) == '1'
MOTION_WATCHER_FPS = float(os.environ.get('POSE_MOTION_WATCHER_FPS', '20'))
MOTION_SMALL_WIDTH = int(os.environ.get('POSE_MOTION_SMALL_WIDTH', '160'))
MOTION_SMALL_HEIGHT = int(os.environ.get('POSE_MOTION_SMALL_HEIGHT', '90'))
MOTION_PIXEL_THRESHOLD = int(os.environ.get('POSE_MOTION_PIXEL_THRESHOLD', '22'))
MOTION_RATIO_THRESHOLD = float(os.environ.get('POSE_MOTION_RATIO_THRESHOLD', '0.018'))
MOTION_MAX_RATIO = float(os.environ.get('POSE_MOTION_MAX_RATIO', '0.70'))
MOTION_CONSECUTIVE_HITS = int(os.environ.get('POSE_MOTION_CONSECUTIVE_HITS', '2'))
MOTION_BURST_HOLD_SEC = float(os.environ.get('POSE_MOTION_BURST_HOLD_SEC', '3.0'))
EDGE_SIGNAL_ENABLED = os.environ.get("POSE_EDGE_SIGNAL", "1") == "1"
EDGE_SIGNAL_WAKE_SCHEDULER = os.environ.get("POSE_EDGE_WAKE_SCHEDULER", "1") == "1"
EDGE_SIGNAL_URL = os.environ.get("POSE_EDGE_SIGNAL_URL", "http://127.0.0.1:8020")
EDGE_SIGNAL_TOKEN_FILE = Path(os.environ.get(
    "POSE_EDGE_SIGNAL_TOKEN_FILE", PROJECT_ROOT / "runtime_data/edge_control/api_token"
))
EDGE_SIGNAL_MAX_AGE_SEC = float(os.environ.get("POSE_EDGE_SIGNAL_MAX_AGE_SEC", "4.0"))
EDGE_MANAGED_CAMERAS = parse_camera_ids(os.environ.get(
    "POSE_EDGE_MANAGED_CAMERAS", "bed_161"
))
EDGE_FAILOVER_GRACE_SEC = float(os.environ.get(
    "POSE_EDGE_FAILOVER_GRACE_SEC", "3.0"
))
EDGE_MANAGED_EMPTY_PROBE_HZ = float(os.environ.get(
    "POSE_EDGE_MANAGED_EMPTY_PROBE_HZ", "0.05"
))
PRE_EVENT_DURATION_SEC = float(os.environ.get('POSE_PRE_EVENT_SECONDS', '10.0'))
PRE_EVENT_SAMPLE_HZ = float(os.environ.get('POSE_PRE_EVENT_HZ', '20.0'))
PRE_EVENT_FRAME_WIDTH = int(os.environ.get('POSE_PRE_EVENT_FRAME_WIDTH', '640'))
PRE_EVENT_JPEG_QUALITY = int(os.environ.get('POSE_PRE_EVENT_JPEG_QUALITY', '70'))
PRE_EVENT_REPLAY_ENABLED = os.environ.get('POSE_PRE_EVENT_REPLAY', '0') == '1'
PRE_EVENT_REPLAY_DURATION_SEC = float(os.environ.get(
    'POSE_PRE_EVENT_REPLAY_SECONDS', '8.0'
))
PRE_EVENT_REPLAY_MAX_FRAMES = int(os.environ.get(
    'POSE_PRE_EVENT_REPLAY_MAX_FRAMES', '150'
))
PRE_EVENT_REPLAY_BATCH_SIZE = int(os.environ.get(
    'POSE_PRE_EVENT_REPLAY_BATCH_SIZE', '8'
))
PRE_EVENT_REPLAY_HOLD_SEC = float(os.environ.get(
    'POSE_PRE_EVENT_REPLAY_HOLD_SEC', '5.0'
))
PRE_EVENT_REPLAY_DEADLINE_SEC = float(os.environ.get(
    'POSE_PRE_EVENT_REPLAY_DEADLINE_SEC', '6.0'
))
EMPTY_POSE_PROBE_HZ = float(os.environ.get('POSE_EMPTY_PROBE_HZ', '0.75'))
OCCUPIED_POSE_INTERVAL_SEC = float(os.environ.get(
    'POSE_OCCUPIED_POSE_INTERVAL_SEC', '0.09'
))
_LIVE_TCN_MAX_INTERVAL = os.environ.get('POSE_LIVE_TCN_MAX_INTERVAL_SEC')
LIVE_TCN_MAX_INTERVAL_SEC = (
    float(_LIVE_TCN_MAX_INTERVAL) if _LIVE_TCN_MAX_INTERVAL else None
)
PERSON_TRACK_TTL_SEC = float(os.environ.get('POSE_TRACK_TTL_SEC', '5.0'))
PRIMARY_SWITCH_MARGIN = float(os.environ.get('POSE_PRIMARY_SWITCH_MARGIN', '0.25'))
SHADOW_RECORD_ENABLED = os.environ.get('POSE_SHADOW_RECORD', '1') == '1'
SHADOW_RECORD_DIR = Path(os.environ.get(
    'POSE_SHADOW_RECORD_DIR', PROJECT_ROOT / 'runtime_data/shadow_features'
))
SHADOW_RECORD_INTERVAL_SEC = float(os.environ.get(
    'POSE_SHADOW_RECORD_INTERVAL_SEC', '0.5'
))
TEMPORAL_SESSION_RECORD_ENABLED = os.environ.get(
    'POSE_TEMPORAL_SESSION_RECORD', '1'
) == '1'
TEMPORAL_SESSION_RECORD_DIR = Path(os.environ.get(
    'POSE_TEMPORAL_SESSION_RECORD_DIR',
    PROJECT_ROOT / 'runtime_data/temporal_sessions',
))
TEMPORAL_SESSION_CAMERAS = parse_camera_ids(os.environ.get(
    'POSE_TEMPORAL_SESSION_CAMERAS', 'bed_161'
))
TEMPORAL_SESSION_PRE_SEC = float(os.environ.get(
    'POSE_TEMPORAL_SESSION_PRE_SEC', '10.0'
))
TEMPORAL_SESSION_POST_SEC = float(os.environ.get(
    'POSE_TEMPORAL_SESSION_POST_SEC', '10.0'
))
TEMPORAL_SESSION_MAX_SEC = float(os.environ.get(
    'POSE_TEMPORAL_SESSION_MAX_SEC', '180.0'
))
TEMPORAL_SESSION_MODEL_REARM_SEC = float(os.environ.get(
    'POSE_TEMPORAL_SESSION_MODEL_REARM_SEC', '60.0'
))
FUSION_POLICY_VERSION = "hybrid_v6_direct_rapid_bed_departure"
CALIBRATION_REPORT_PATH = Path(os.environ.get(
    "POSE_CALIBRATION_REPORT", PROJECT_ROOT / "runtime_data/operational_report.json"
))
PROCESS_STARTED_MONO = time.monotonic()

CLASS_NAMES = ['front_lying', 'prone_back', 'side_near', 'side_far', 'sitting_center', 'sitting_edge']


def orient_analysis_frame(frame: np.ndarray) -> np.ndarray:
    """Rotate only the inference coordinate system; raw viewer stays untouched."""
    if ANALYSIS_ROTATION == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if ANALYSIS_ROTATION == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if ANALYSIS_ROTATION == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

class DirectLockedKerasPredictor:
    """Serialize shared Keras inference without ``Model.predict`` overhead.

    ``Model.predict`` builds a data adapter and callback loop for every call.
    For the one-frame 34-value posture classifier that costs tens of
    milliseconds and, because the model is shared by six camera threads,
    prevents the 20 Hz temporal contract from ever becoming ready. The eager
    inference call is equivalent for inference and keeps one shared owner.
    """

    def __init__(self, predictor):
        self._predictor = predictor
        self._lock = Lock()

    def predict(self, inputs, *, verbose=0):
        del verbose
        with self._lock:
            output = self._predictor(inputs, training=False)
            return (
                output.numpy()
                if hasattr(output, "numpy") else np.asarray(output)
            )


class ParallelInferencePool:
    """Own the two YOLO weights used by the central scheduler."""
    def __init__(self, seg_weight, pose_weight, device='0', workers=3):
        self.seg_model = YOLO(seg_weight)
        self.pose_model = YOLO(pose_weight)
        self.bed_refiner = (
            SAM(BED_REFINER_WEIGHT)
            if BED_REFINER_ENABLED and Path(BED_REFINER_WEIGHT).is_file()
            else None
        )
        self.device = device
        self.lock = Lock()
        self.refiner_lock = Lock()

    def infer_seg(self, frame, conf=0.1, classes=[0]):
        with self.lock:
            return self.seg_model.predict(frame, conf=conf, classes=classes, device=self.device, verbose=False)

    def infer_pose(self, frame, conf=0.5):
        with self.lock:
            return self.pose_model.predict(frame, conf=conf, device=self.device, verbose=False)

    def refine_bed(self, frame: np.ndarray, coarse_bed: dict) -> dict | None:
        if self.bed_refiner is None or coarse_bed.get('bbox') is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = coarse_bed['bbox']
        fractions = (0.50, 0.62, 0.72)
        if abs(x2 - x1) >= abs(y2 - y1):
            points = [(
                float(np.clip(x1 + (x2 - x1) * fraction, 0, w - 1)),
                float(np.clip((y1 + y2) / 2.0, 0, h - 1)),
            ) for fraction in fractions]
        else:
            points = [(
                float(np.clip((x1 + x2) / 2.0, 0, w - 1)),
                float(np.clip(y1 + (y2 - y1) * fraction, 0, h - 1)),
            ) for fraction in fractions]
        with self.refiner_lock:
            results = self.bed_refiner.predict(
                frame,
                points=[[x, y] for x, y in points],
                labels=[1] * len(points),
                device=BED_REFINER_DEVICE,
                verbose=False,
            )
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return None
        masks = [
            _mask_to_frame(mask_tensor, h, w)
            for mask_tensor in results[0].masks.data
        ]
        return select_refined_bed_candidate(
            masks,
            points,
            coarse_bed.get('bbox'),
            float(coarse_bed.get('confidence', 0.0)),
            h,
            w,
            min_area_ratio=BED_REFINER_MIN_AREA_RATIO,
            max_area_ratio=BED_REFINER_MAX_AREA_RATIO,
            min_extent_ratio=BED_REFINER_MIN_EXTENT_RATIO,
        )

    def shutdown(self):
        pass

def _mask_to_frame(mask_tensor, h: int, w: int) -> np.ndarray:
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[0]
    mask_np = mask_tensor.cpu().numpy()
    if mask_np.shape != (h, w):
        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
    return (mask_np * 255).astype(np.uint8)

def extract_bed_detection(seg_result, h: int, w: int) -> dict:
    bed = {'mask': None, 'bbox': None, 'confidence': 0.0, 'source': 'none'}
    if seg_result is None or seg_result.boxes is None or len(seg_result.boxes) == 0:
        return bed
    best_idx = int(seg_result.boxes.conf.argmax())
    x1, y1, x2, y2 = seg_result.boxes.xyxy[best_idx].cpu().numpy()
    bed['bbox'] = (int(x1), int(y1), int(x2), int(y2))
    bed['confidence'] = float(seg_result.boxes.conf[best_idx].item())
    if seg_result.masks is not None and len(seg_result.masks) > best_idx:
        bed['mask'] = _mask_to_frame(seg_result.masks.data[best_idx], h, w)
        bed['source'] = 'seg_mask'
    else:
        bed['source'] = 'seg_bbox'
    return bed


def compute_body_in_bed_ratio(
        kpts_xy: np.ndarray,
        kpts_conf: np.ndarray | None,
        bed: dict,
        frame_h: int,
        frame_w: int,
        min_kpt_conf: float = 0.3,
) -> float:
        return skeleton_bed_coverage(
            kpts_xy, kpts_conf, bed, frame_h, frame_w, min_kpt_conf
        )


def extract_pose_detections(
    pose_result,
    bed: dict,
    frame_h: int,
    frame_w: int,
) -> list[PersonDetection]:
    """Convert one Ultralytics result into the shared tracker contract."""
    detections: list[PersonDetection] = []
    if pose_result is None:
        return detections
    pose_keypoints = getattr(pose_result, "keypoints", None)
    if pose_keypoints is None or len(pose_keypoints) == 0:
        return detections
    all_xy = pose_keypoints.xy.cpu().numpy()
    if pose_keypoints.conf is not None:
        all_conf = pose_keypoints.conf.cpu().numpy()
    else:
        all_conf = np.ones((len(all_xy), 17), dtype=np.float32)
    pose_boxes = getattr(pose_result, "boxes", None)
    box_confidences = (
        pose_boxes.conf.cpu().numpy()
        if pose_boxes is not None and pose_boxes.conf is not None
        else np.ones(len(all_xy), dtype=np.float32)
    )
    box_coordinates = (
        pose_boxes.xyxy.cpu().numpy()
        if pose_boxes is not None and pose_boxes.xyxy is not None
        else None
    )
    for detection_idx in range(len(all_xy)):
        candidate_xy = np.asarray(all_xy[detection_idx], dtype=np.float32)
        candidate_conf = np.asarray(all_conf[detection_idx], dtype=np.float32)
        box_confidence = float(box_confidences[detection_idx])
        visible_count = int(np.count_nonzero(candidate_conf >= 0.2))
        keypoint_mean = float(np.mean(candidate_conf)) if candidate_conf.size else 0.0
        area_ratio = 0.0
        if box_coordinates is not None:
            bx1, by1, bx2, by2 = box_coordinates[detection_idx]
            area_ratio = float(
                max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                / max(1.0, float(frame_h * frame_w))
            )
        # A high detector score alone is insufficient in the fisheye rooms:
        # compact static equipment can resemble a tiny articulated person.
        # Every accepted person must occupy a meaningful fraction of frame.
        if not accept_pose_candidate(
            box_confidence=box_confidence,
            area_ratio=area_ratio,
            visible_count=visible_count,
            keypoint_mean=keypoint_mean,
            strong_box_conf=POSE_STRONG_BOX_CONF,
            strong_min_area_ratio=POSE_STRONG_MIN_AREA_RATIO,
            weak_box_conf=POSE_WEAK_BOX_CONF,
            weak_min_area_ratio=POSE_WEAK_MIN_AREA_RATIO,
            weak_min_visible=POSE_WEAK_MIN_VISIBLE,
            weak_min_keypoint_mean=POSE_WEAK_MIN_KP_MEAN,
        ):
            continue
        keypoint_box = keypoints_bbox(candidate_xy, candidate_conf)
        detector_box = (
            box_coordinates[detection_idx]
            if box_coordinates is not None else None
        )
        bbox = select_tracking_bbox(detector_box, keypoint_box)
        if bbox is None:
            continue
        bed_overlap = compute_body_in_bed_ratio(
            candidate_xy, candidate_conf, bed, frame_h, frame_w
        )
        visible = candidate_conf[candidate_conf >= 0.2]
        confidence = float(np.mean(visible)) if len(visible) else 0.0
        detections.append(PersonDetection(
            keypoints_xy=candidate_xy,
            keypoints_conf=candidate_conf,
            bbox=bbox,
            confidence=confidence,
            bed_overlap=bed_overlap,
        ))
    return detections


def scale_bed_geometry(
    bed: dict, *, source_h: int, source_w: int,
    target_h: int, target_w: int,
) -> dict:
    """Scale cached bed geometry to a compressed pre-roll frame."""
    scaled = dict(bed or {})
    mask = scaled.get("mask")
    if mask is not None and mask.shape[:2] != (target_h, target_w):
        scaled["mask"] = cv2.resize(
            mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
    bbox = scaled.get("bbox")
    if bbox is not None and source_w > 0 and source_h > 0:
        sx = target_w / float(source_w)
        sy = target_h / float(source_h)
        x1, y1, x2, y2 = bbox
        scaled["bbox"] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    return scaled


def empty_replay_status(threshold: float = 0.0, *, reason: str = "not_started") -> dict:
    return {
        "ready": False, "probability": 0.0, "candidate": False,
        "threshold": float(threshold), "samples": 0, "prediction_count": 0,
        "requested_frames": 0, "observed_pose_frames": 0,
        "track_reset_total": 0, "source": "none", "reason": reason,
        "elapsed_ms": 0.0,
    }


def run_pre_event_replay(
    camera_id: str,
    watcher: MotionWatcher,
    scheduler: LatestInferenceScheduler,
    keras_clf,
    temporal_service: TemporalModelService,
    bed: dict,
    *,
    source_h: int,
    source_w: int,
    trigger_mono_ts: float,
) -> dict:
    """Run one bounded batch replay for a motion trigger."""
    started = time.perf_counter()
    items = watcher.pre_event_snapshot(
        end_mono_ts=trigger_mono_ts,
        duration_sec=PRE_EVENT_REPLAY_DURATION_SEC,
        max_frames=PRE_EVENT_REPLAY_MAX_FRAMES,
    )
    decoded: list[tuple[object, np.ndarray]] = []
    for item in items:
        frame = item.decode()
        if frame is not None:
            decoded.append((item, orient_analysis_frame(frame)))
    if len(decoded) < 30:
        status = empty_replay_status(
            temporal_service.threshold, reason="fewer_than_30_decoded_frames"
        )
        status["requested_frames"] = len(decoded)
        return status

    pose_results = []
    batch_size = max(1, PRE_EVENT_REPLAY_BATCH_SIZE)
    for offset in range(0, len(decoded), batch_size):
        elapsed = time.perf_counter() - started
        remaining = PRE_EVENT_REPLAY_DEADLINE_SEC - elapsed
        if remaining <= 0.05:
            status = empty_replay_status(
                temporal_service.threshold, reason="total_budget_exceeded"
            )
            status["requested_frames"] = len(decoded)
            return status
        chunk = decoded[offset:offset + batch_size]
        outcome = scheduler.request_pose_replay(
            camera_id,
            [frame for _, frame in chunk],
            frame_seq=chunk[-1][0].frame_seq,
            priority=P3_EMPTY_PROBE,
            deadline_sec=min(2.0, remaining),
        )
        if not outcome.completed:
            status = empty_replay_status(
                temporal_service.threshold,
                reason=f"scheduler_{outcome.drop_reason or 'not_completed'}",
            )
            status["requested_frames"] = len(decoded)
            return status
        pose_results.extend(list(outcome.result or []))
    if len(pose_results) != len(decoded):
        status = empty_replay_status(
            temporal_service.threshold, reason="pose_result_count_mismatch"
        )
        status["requested_frames"] = len(decoded)
        return status

    replay_frames: list[ReplayPoseFrame] = []
    for (item, frame), result in zip(decoded, pose_results):
        rh, rw = frame.shape[:2]
        replay_bed = scale_bed_geometry(
            bed, source_h=source_h, source_w=source_w,
            target_h=rh, target_w=rw,
        )
        detections = extract_pose_detections(result, replay_bed, rh, rw)
        replay_frames.append(ReplayPoseFrame(
            mono_ts=item.mono_ts,
            frame_width=rw,
            frame_height=rh,
            detections=tuple(detections),
        ))

    status = replay_temporal_context(
        replay_frames,
        temporal_service,
        lambda batch: keras_clf.predict(batch, verbose=0),
        track_ttl_sec=1.0,
        primary_switch_margin=PRIMARY_SWITCH_MARGIN,
    )
    status["reason"] = "completed"
    status["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    return status


# Optimized mask overlay helper: downscale mask, find contours, draw scaled contours on frame
def draw_mask_contours(mask: np.ndarray, frame: np.ndarray, downscale:int=2, color=(60,200,120), alpha:float=0.22) -> np.ndarray:
    """Draw mask contours on frame using downscaling to reduce CPU.
    mask: uint8 mask (0..255) same size as frame or smaller
    frame: BGR image
    """
    try:
        fh, fw = frame.shape[:2]
        m = mask
        # ensure binary
        if m.max() > 1 and m.max() <= 255:
            _, m_bin = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
        else:
            m_bin = (m > 0).astype('uint8') * 255
        # downscale
        if downscale > 1:
            small = cv2.resize(m_bin, (max(1, fw//downscale), max(1, fh//downscale)), interpolation=cv2.INTER_NEAREST)
        else:
            small = m_bin
        # find contours on small mask
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame
        # scale contours up
        scaled_contours = []
        sx = fw / small.shape[1]
        sy = fh / small.shape[0]
        for c in contours:
            c = c.reshape(-1,2).astype('float32')
            c[:,0] = np.clip(np.round(c[:,0] * sx).astype(int), 0, fw-1)
            c[:,1] = np.clip(np.round(c[:,1] * sy).astype(int), 0, fh-1)
            scaled_contours.append(c.reshape(-1,1,2).astype('int32'))
        overlay = frame.copy()
        # fill contours
        cv2.drawContours(overlay, scaled_contours, -1, color, thickness=cv2.FILLED)
        out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return out
    except Exception as e:
        logging.warning(f"draw_mask_contours error: {e}")
        return frame

class CameraState(BaseModel):
    camera_id: str = "none"
    camera_name: str = "none"
    in_bed: str = "NO"
    pose: str = "None"
    pose_conf: float = 0.0
    fall_score: float = 0.0
    fall_level: str = "SAFE"
    fall_status: str = "NO_PERSON"
    seg_attachment: str = "none"
    risk_level: str = "SAFE"
    bed_event: str | None = None
    timestamp: str = ""
    latency_ms: float = 0.0
    pipeline_fps: float = 0.0
    status: str = "disconnected"
    analysis_state: str = "idle"  # idle, detecting, analyzing, tracking
    body_in_bed_ratio: float = 0.0
    tcn_shadow_enabled: bool = False
    tcn_shadow_ready: bool = False
    tcn_fall_probability: float = 0.0
    tcn_alert_candidate: bool = False
    tcn_threshold: float = 0.0
    tcn_samples: int = 0
    tcn_prediction_count: int = 0
    tcn_sample_hz: float = 10.0
    tcn_window_rows: int = 30
    tcn_fusion_enabled: bool = TCN_FUSION_ENABLED
    tcn_missing_samples_window: int = 0
    tcn_missing_samples_total: int = 0
    tcn_last_sample_timestamp: float | None = None
    tcn_source: str = "none"
    tcn_replay_enabled: bool = PRE_EVENT_REPLAY_ENABLED
    tcn_replay_ready: bool = False
    tcn_replay_probability: float = 0.0
    tcn_replay_candidate: bool = False
    tcn_replay_samples: int = 0
    tcn_replay_requested_frames: int = 0
    tcn_replay_observed_frames: int = 0
    tcn_replay_attempt_total: int = 0
    tcn_replay_completed_total: int = 0
    tcn_replay_error_total: int = 0
    tcn_replay_elapsed_ms: float = 0.0
    tcn_replay_reason: str = "not_started"
    tcn_replay_valid_remaining_ms: float = 0.0
    tcn_replay_gap_reset_total: int = 0
    capture_connected: bool = False
    capture_fps: float = 0.0
    capture_frame_seq: int = 0
    capture_frame_age_ms: float | None = None
    capture_decode_error_total: int = 0
    capture_reconnect_total: int = 0
    analysis_frame_seq: int = 0
    analysis_frame_age_ms: float | None = None
    bed_roi_ready: bool = False
    roi_state: str = "NOT_READY"
    bed_roi_restored_from_cache: bool = False
    bed_roi_version: int = 0
    bed_roi_source: str = "auto_not_ready"
    bed_roi_confidence: float = 0.0
    bed_roi_agreement_iou: float = 0.0
    bed_roi_candidate_count: int = 0
    bed_seg_run_count: int = 0
    bed_roi_invalid_reason: str = "no_auto_roi"
    runtime_mode: str = "EMPTY"
    watcher_fps: float = 0.0
    watcher_frame_seq: int = 0
    watcher_processed_total: int = 0
    watcher_thread_alive: bool = False
    motion_ratio: float = 0.0
    motion_detected: bool = False
    motion_hit_streak: int = 0
    motion_trigger_total: int = 0
    burst_active: bool = False
    burst_remaining_ms: float = 0.0
    edge_signal_connected: bool = False
    edge_signal_wake_active: bool = False
    edge_signal_result_fresh: bool = False
    edge_signal_result_age_ms: float | None = None
    edge_signal_person_present: bool = False
    edge_signal_last_person_present: bool = False
    edge_signal_pose_confidence: float = 0.0
    edge_signal_quality: float = 0.0
    edge_signal_frame_seq: int | None = None
    edge_signal_model_bundle_version: str | None = None
    edge_signal_error_total: int = 0
    edge_managed: bool = False
    edge_local_watcher_suppressed: bool = False
    edge_fallback_active: bool = False
    edge_effective_empty_probe_hz: float = EMPTY_POSE_PROBE_HZ
    pre_event_frames: int = 0
    pre_event_coverage_sec: float = 0.0
    pre_event_bytes: int = 0
    pre_event_target_sec: float = PRE_EVENT_DURATION_SEC
    pre_event_sample_hz: float = PRE_EVENT_SAMPLE_HZ
    pre_event_ready: bool = False
    pose_inference_total: int = 0
    pose_inference_fps: float = 0.0
    scheduler_completed_total: int = 0
    scheduler_completed_hz: float = 0.0
    scheduler_queue_latency_ms: float = 0.0
    scheduler_inference_ms: float = 0.0
    scheduler_stale_drop_total: int = 0
    scheduler_superseded_drop_total: int = 0
    scheduler_timeout_total: int = 0
    scheduler_error_total: int = 0
    scheduler_pending: int = 0
    scheduler_last_priority: int | None = None
    scheduler_last_model: str | None = None
    scheduler_thread_alive: bool = False
    person_count: int = 0
    track_count: int = 0
    primary_track_id: int | None = None
    track_switch_total: int = 0
    track_created_total: int = 0
    track_expired_total: int = 0
    primary_track_observed: bool = False
    primary_track_bed_overlap: float = 0.0
    primary_track_confidence: float = 0.0
    tcn_track_id: int | None = None
    tcn_track_reset_total: int = 0
    tcn_gap_reset_total: int = 0
    tcn_duplicate_skip_total: int = 0
    tcn_non_monotonic_skip_total: int = 0
    tcn_last_dt_sec: float | None = None
    tcn_last_action: str = "not_started"
    fusion_phase: str = "NO_PERSON"
    fusion_risk: float = 0.0
    fusion_evidence: list[str] = []
    fusion_safe_evidence: list[str] = []
    fusion_candidate_age_sec: float = 0.0
    fusion_quality: float = 0.0
    fusion_track_id: int | None = None
    fusion_policy_version: str = FUSION_POLICY_VERSION
    feature_recorder_enabled: bool = False
    feature_recorder_thread_alive: bool = False
    feature_recorder_written_total: int = 0
    feature_recorder_dropped_total: int = 0
    feature_recorder_error_total: int = 0
    feature_recorder_queue_depth: int = 0

state_lock = Lock()
camera_states = {cid: CameraState(camera_id=cid, camera_name=cfg['name']) for cid, cfg in CAMERA_CONFIGS.items()}
camera_captures = {}
bed_roi_managers = {}
motion_watchers = {}
analysis_running = False
inference_pool = None
inference_scheduler = None
temporal_service = None
shadow_recorder = None
temporal_session_recorder = None
edge_signal_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analysis_running, inference_pool, inference_scheduler, temporal_service
    global shadow_recorder, temporal_session_recorder
    global edge_signal_client
    global camera_captures, bed_roi_managers, motion_watchers
    logging.info(f"[🚀] Multi-Camera Pipeline (Unified Logic) 시작")
    edge_signal_client = None
    if EDGE_SIGNAL_ENABLED and EDGE_SIGNAL_TOKEN_FILE.is_file():
        edge_signal_client = EdgeSignalClient(
            EDGE_SIGNAL_URL, EDGE_SIGNAL_TOKEN_FILE,
            max_result_age_sec=EDGE_SIGNAL_MAX_AGE_SEC,
        )
        edge_signal_client.start()
        logging.info(
            f"[edge signal] polling {EDGE_SIGNAL_URL}; "
            f"scheduler_wake={int(EDGE_SIGNAL_WAKE_SCHEDULER)}"
        )
    inference_pool = ParallelInferencePool(YOLO_SEG_WEIGHT, YOLO_POSE_WEIGHT, device=YOLO_DEVICE, workers=PARALLEL_WORKERS)
    inference_scheduler = LatestInferenceScheduler(
        lambda frame: inference_pool.infer_pose(frame, conf=PERSON_POSE_CONF),
        lambda frame: inference_pool.infer_seg(
            frame, conf=BED_SEG_CONF, classes=[YOLO_SEG_CLASS]
        ),
        infer_pose_replay=lambda frame: inference_pool.infer_pose(
            frame, conf=REPLAY_PERSON_POSE_CONF
        ),
        urgent_quota=INFERENCE_URGENT_QUOTA,
    )
    inference_scheduler.start()
    shadow_recorder = None
    if SHADOW_RECORD_ENABLED:
        shadow_recorder = ShadowFeatureRecorder(
            SHADOW_RECORD_DIR,
            sample_interval_sec=SHADOW_RECORD_INTERVAL_SEC,
        )
        shadow_recorder.start()
        logging.info(
            f"[shadow recorder] feature-only {SHADOW_RECORD_INTERVAL_SEC:.2f}s "
            f"-> {SHADOW_RECORD_DIR}"
        )
    temporal_session_recorder = None
    if TEMPORAL_SESSION_RECORD_ENABLED:
        temporal_session_recorder = TemporalEventSessionRecorder(
            TEMPORAL_SESSION_RECORD_DIR,
            pre_roll_sec=TEMPORAL_SESSION_PRE_SEC,
            post_roll_sec=TEMPORAL_SESSION_POST_SEC,
            max_session_sec=TEMPORAL_SESSION_MAX_SEC,
            model_trigger_rearm_sec=TEMPORAL_SESSION_MODEL_REARM_SEC,
        )
        temporal_session_recorder.start()
        logging.info(
            "[temporal session recorder] automatic triggers; cameras=%s; "
            "pre=%.1fs post=%.1fs -> %s",
            sorted(TEMPORAL_SESSION_CAMERAS),
            TEMPORAL_SESSION_PRE_SEC,
            TEMPORAL_SESSION_POST_SEC,
            TEMPORAL_SESSION_RECORD_DIR,
        )
    # Keep latency-sensitive one-frame classification independent from the
    # much larger replay batch.  Each model is serialized within its own use
    # class, so replay can never hold the live Keras lock.
    keras_clf = DirectLockedKerasPredictor(
        keras.models.load_model(POSE_KERAS_MODEL)
    )
    keras_replay_clf = (
        DirectLockedKerasPredictor(
            keras.models.load_model(POSE_KERAS_MODEL)
        )
        if PRE_EVENT_REPLAY_ENABLED else keras_clf
    )
    temporal_service = None
    if TCN_SHADOW_ENABLED:
        try:
            threshold = float(TCN_THRESHOLD) if TCN_THRESHOLD is not None else None
            temporal_service = TemporalModelService(
                TCN_MODEL_PATH, TCN_REPORT_PATH, device=TCN_DEVICE,
                threshold=threshold,
                allow_non_promotion=TCN_ALLOW_NON_PROMOTION,
            )
            logging.info(
                f"[TCN shadow] loaded {TCN_MODEL_PATH} on {TCN_DEVICE}; "
                f"threshold={temporal_service.threshold:.4f}"
            )
        except Exception as exc:
            logging.exception(f"[TCN shadow] disabled: {exc}")
    analysis_running = True
    camera_captures = {
        cid: LatestFrameCapture(
            cid,
            cfg["rtsp_url"],
            frame_width=FRAME_WIDTH,
        )
        for cid, cfg in CAMERA_CONFIGS.items()
    }
    bed_roi_managers = {
        cid: AutoBedROIManager(
            cid,
            AUTO_BED_CACHE_DIR,
            sample_every_n=SEG_EVERY_N,
            candidate_window=AUTO_BED_CANDIDATE_WINDOW,
            min_detections=AUTO_BED_MIN_DETECTIONS,
            consensus_iou=AUTO_BED_CONSENSUS_IOU,
            refresh_sec=AUTO_BED_REFRESH_SEC,
            scene_change_ratio=AUTO_BED_SCENE_CHANGE_RATIO,
            scene_change_persistence=AUTO_BED_SCENE_CHANGE_PERSISTENCE,
        )
        for cid in CAMERA_CONFIGS
    }
    motion_watchers = {
        cid: MotionWatcher(
            cid,
            camera_captures[cid],
            roi_provider=lambda manager=bed_roi_managers[cid]: manager.current_bbox(),
            target_fps=MOTION_WATCHER_FPS,
            small_width=MOTION_SMALL_WIDTH,
            small_height=MOTION_SMALL_HEIGHT,
            pixel_threshold=MOTION_PIXEL_THRESHOLD,
            motion_ratio_threshold=MOTION_RATIO_THRESHOLD,
            max_motion_ratio=MOTION_MAX_RATIO,
            consecutive_hits=MOTION_CONSECUTIVE_HITS,
            burst_hold_sec=MOTION_BURST_HOLD_SEC,
            pre_event_duration_sec=PRE_EVENT_DURATION_SEC,
            pre_event_sample_hz=PRE_EVENT_SAMPLE_HZ,
            pre_event_frame_width=PRE_EVENT_FRAME_WIDTH,
            pre_event_jpeg_quality=PRE_EVENT_JPEG_QUALITY,
        )
        for cid in CAMERA_CONFIGS
    }
    for capture in camera_captures.values():
        capture.start()
    for cid, watcher in motion_watchers.items():
        if cid not in EDGE_MANAGED_CAMERAS:
            watcher.start()

    threads = []
    for cid in CAMERA_CONFIGS.keys():
        t = Thread(
            target=run_analysis,
            args=(
                cid, camera_captures[cid], inference_scheduler, keras_clf,
                keras_replay_clf, temporal_service,
                bed_roi_managers[cid], motion_watchers[cid],
                edge_signal_client, temporal_session_recorder,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)
    try:
        yield
    finally:
        analysis_running = False
        for watcher in motion_watchers.values():
            watcher.stop()
        if edge_signal_client is not None:
            edge_signal_client.stop()
        for capture in camera_captures.values():
            capture.stop()
        for thread in threads:
            thread.join(timeout=3.0)
        if temporal_session_recorder is not None:
            temporal_session_recorder.stop()
        inference_scheduler.stop()
        if shadow_recorder is not None:
            shadow_recorder.stop()
        inference_pool.shutdown()

app = FastAPI(lifespan=lifespan)

def run_analysis(
    camera_id: str,
    capture: LatestFrameCapture,
    scheduler: LatestInferenceScheduler,
    keras_clf,
    keras_replay_clf,
    temporal_service=None,
    bed_roi_manager: AutoBedROIManager | None = None,
    motion_watcher: MotionWatcher | None = None,
    signal_client: EdgeSignalClient | None = None,
    session_recorder: TemporalEventSessionRecorder | None = None,
):
    global camera_states, analysis_running
    # [RULE] 원본 아키텍처 상태 객체
    preset = load_preset()
    motion_state = MotionState()
    event_tracker = LiveEventTracker(preset)
    scoring_cfg = preset.get("scoring") or {}
    fall_scorer = FallScorer(scoring_cfg) if scoring_cfg.get("enabled") else None

    # State machine + temporal buffer. Motion is watched by a separate
    # high-frequency CPU thread, never by this slower AI loop.
    state_machine = CameraAnalysisStateMachine(buffer_ready_threshold=3, idle_timeout_sec=2.0)
    # Person identity owns temporal history; camera/detection order never does.
    person_tracker = MultiPersonTracker(
        track_ttl_sec=PERSON_TRACK_TTL_SEC,
        primary_switch_margin=PRIMARY_SWITCH_MARGIN,
    )
    hybrid_fusion = HybridFusion(
        motion_ratio_threshold=MOTION_RATIO_THRESHOLD,
    )
    frame_buffers: dict[int, FrameBuffer] = {}
    temporal_runners: dict[int, TemporalShadowRunner] = {}
    tcn_track_reset_total = 0

    frame_idx = 0
    if bed_roi_manager is None:
        bed_roi_manager = AutoBedROIManager(
            camera_id, AUTO_BED_CACHE_DIR, sample_every_n=SEG_EVERY_N,
            candidate_window=AUTO_BED_CANDIDATE_WINDOW,
            min_detections=AUTO_BED_MIN_DETECTIONS,
            consensus_iou=AUTO_BED_CONSENSUS_IOU,
            refresh_sec=AUTO_BED_REFRESH_SEC,
            scene_change_ratio=AUTO_BED_SCENE_CHANGE_RATIO,
            scene_change_persistence=AUTO_BED_SCENE_CHANGE_PERSISTENCE,
        )
    frame_times = []
    pose_inference_times = []
    pose_inference_total = 0
    next_empty_pose_probe = 0.0
    edge_disconnected_since: float | None = None
    last_pose_request_capture_ts: float | None = None
    verify_priority_until = 0.0
    last_capture_seq = 0
    last_replay_trigger_total = 0
    replay_attempt_total = 0
    replay_completed_total = 0
    replay_error_total = 0
    replay_valid_until = 0.0
    next_presence_replay_at = 0.0
    next_replay_allowed_at = 0.0
    replay_status = empty_replay_status(
        temporal_service.threshold if temporal_service is not None else 0.0
    )
    replay_worker = AsyncReplayWorker[dict](camera_id)
    previous_session_state = {
        "person_observed": False,
        "edge_wake": False,
        "local_motion": False,
        "tcn_candidate": False,
        "fusion_phase": "NO_PERSON",
        "track_id": None,
    }

    while analysis_running:
        replay_completion = replay_worker.poll()
        if replay_completion is not None:
            if replay_completion.error is not None:
                replay_error_total += 1
                replay_valid_until = 0.0
                replay_status = empty_replay_status(
                    temporal_service.threshold if temporal_service is not None else 0.0,
                    reason=f"async_error_{type(replay_completion.error).__name__}",
                )
                logging.error(
                    "[%s] async pre-event replay failed: %s",
                    camera_id, replay_completion.error,
                )
            else:
                replay_status = replay_completion.value or empty_replay_status(
                    temporal_service.threshold if temporal_service is not None else 0.0,
                    reason="async_empty_result",
                )
                if replay_status.get("reason") == "completed":
                    replay_completed_total += 1
                else:
                    replay_error_total += 1
                replay_valid_until = (
                    time.monotonic() + PRE_EVENT_REPLAY_HOLD_SEC
                    if replay_status.get("ready") else 0.0
                )
        packet = capture.wait_for_frame(last_capture_seq, timeout=1.0)
        if packet is None:
            capture_status = capture.metrics()
            if not capture_status["connected"]:
                with state_lock:
                    camera_states[camera_id].status = "disconnected"
            continue
        last_capture_seq = packet.frame_seq
        frame = orient_analysis_frame(packet.frame)

        loop_start = time.perf_counter()
        fh, fw = frame.shape[:2]
        frame_idx += 1

        # 1. Automatic bed discovery. Segmentation runs only while the
        # camera is learning its bed, after a scene invalidation, or at refresh.
        bed_roi_manager.observe_frame(frame, packet.capture_mono_ts)
        run_bed_seg = bed_roi_manager.should_run_segmentation(
            frame_idx, packet.capture_mono_ts
        )
        seg_res = None
        if run_bed_seg:
            bed_roi_manager.mark_segmentation_attempt(packet.capture_mono_ts)
            # Bed Seg was trained in the native camera orientation. Rotate only
            # the resulting geometry into the Pose coordinate system.
            raw_frame = packet.frame
            raw_h, raw_w = raw_frame.shape[:2]
            seg_outcome = scheduler.request_seg(
                camera_id, raw_frame, frame_seq=packet.frame_seq,
                priority=P4_BED_SEG, deadline_sec=2.0,
            )
            seg_res = seg_outcome.result if seg_outcome.completed else None
            if seg_res is not None and len(seg_res) > 0:
                coarse_bed = extract_bed_detection(seg_res[0], raw_h, raw_w)
                fresh_bed = inference_pool.refine_bed(raw_frame, coarse_bed)
                if fresh_bed is not None:
                    fresh_bed = orient_bed_detection(
                        fresh_bed, ANALYSIS_ROTATION, raw_h, raw_w
                    )
                    bed_roi_manager.observe_detection(
                        fresh_bed,
                        frame,
                        mono_ts=packet.capture_mono_ts,
                        wall_ts=packet.capture_wall_ts,
                    )

        # 2. Suppress the duplicate central watcher while an authenticated
        # edge node is healthy. RTSP capture/viewing remains active. A bounded
        # connection-loss grace restores the central watcher automatically.
        policy_now_mono = time.monotonic()
        edge_signal_status = (
            signal_client.status(camera_id) if signal_client is not None else {
                "connected": False, "wake_active": False,
                "result_fresh": False, "result_age_ms": None,
                "person_present": False, "pose_confidence": 0.0,
                "last_person_present": False,
                "quality": 0.0, "frame_seq": None,
                "model_bundle_version": None, "error_total": 0,
            }
        )
        edge_managed = camera_id in EDGE_MANAGED_CAMERAS
        # The control API being reachable is not enough: a stale node result
        # cannot wake this camera. Treat it as unhealthy and restore the
        # central watcher after the same bounded failover grace.
        edge_signal_healthy = bool(
            edge_signal_status["connected"]
            and edge_signal_status["result_fresh"]
        )
        if edge_managed and not edge_signal_healthy:
            if edge_disconnected_since is None:
                edge_disconnected_since = policy_now_mono
        else:
            edge_disconnected_since = None
        disconnected_for_sec = (
            0.0 if edge_disconnected_since is None
            else max(0.0, policy_now_mono - edge_disconnected_since)
        )
        edge_policy = edge_runtime_policy(
            managed=edge_managed,
            connected=bool(edge_signal_status["connected"]),
            result_fresh=bool(edge_signal_status["result_fresh"]),
            disconnected_for_sec=disconnected_for_sec,
            failover_grace_sec=EDGE_FAILOVER_GRACE_SEC,
            normal_empty_probe_hz=EMPTY_POSE_PROBE_HZ,
            managed_empty_probe_hz=EDGE_MANAGED_EMPTY_PROBE_HZ,
        )
        if motion_watcher is not None:
            if edge_policy["suppress_local_watcher"]:
                if motion_watcher.is_alive():
                    motion_watcher.stop(timeout=0.25)
            elif not motion_watcher.is_alive():
                motion_watcher.start()

        watcher_status = motion_watcher.status() if motion_watcher is not None else {
            "watcher_fps": 0.0, "motion_ratio": 0.0,
            "motion_detected": False, "motion_hit_streak": 0,
            "burst_active": False, "burst_remaining_ms": 0.0,
            "watcher_frame_seq": 0, "watcher_processed_total": 0,
            "motion_trigger_total": 0, "watcher_thread_alive": False,
            "pre_event_frames": 0, "pre_event_coverage_sec": 0.0,
            "pre_event_bytes": 0, "pre_event_target_sec": PRE_EVENT_DURATION_SEC,
            "pre_event_sample_hz": PRE_EVENT_SAMPLE_HZ, "pre_event_ready": False,
        }
        local_burst_active = bool(watcher_status["burst_active"])
        edge_wake_active = bool(edge_signal_status["wake_active"])
        burst_active = local_burst_active or (
            EDGE_SIGNAL_WAKE_SCHEDULER and edge_wake_active
        )

        # 3. Bed zone is usable only after automatic multi-frame consensus.
        # There is deliberately no manual ROI fallback.
        bed = bed_roi_manager.current()
        if preset.get("bed_zone"):
            bed = build_approx_bed_zone(bed, None, fh, fw, preset)
        bed_roi_status = bed_roi_manager.status()

        # 3.5 A new motion episode may need history that the low-rate live Pose
        # path never observed. Replay compressed pre-roll into an isolated
        # tracker/TCN, bounded by frame and wall-clock budgets.
        trigger_total = int(watcher_status.get("motion_trigger_total", 0))
        if trigger_total > last_replay_trigger_total:
            last_replay_trigger_total = trigger_total
            live_context_ready = any(
                runner.status().get("ready", False)
                for runner in temporal_runners.values()
            )
            if (
                PRE_EVENT_REPLAY_ENABLED
                and temporal_service is not None
                and motion_watcher is not None
                and bool(watcher_status.get("pre_event_ready", False))
                and not live_context_ready
                and time.monotonic() >= next_replay_allowed_at
            ):
                trigger_ts = watcher_status.get("last_motion_trigger_mono_ts")
                replay_bed = dict(bed or {})
                if replay_bed.get("mask") is not None:
                    replay_bed["mask"] = replay_bed["mask"].copy()
                replay_trigger_ts = float(
                    packet.capture_mono_ts if trigger_ts is None else trigger_ts
                )
                submitted = replay_worker.submit(lambda
                    replay_bed=replay_bed,
                    replay_source_h=fh,
                    replay_source_w=fw,
                    replay_trigger_ts=replay_trigger_ts:
                    run_pre_event_replay(
                        camera_id, motion_watcher, scheduler, keras_replay_clf,
                        temporal_service, replay_bed,
                        source_h=replay_source_h,
                        source_w=replay_source_w,
                        trigger_mono_ts=replay_trigger_ts,
                    )
                )
                if submitted:
                    replay_attempt_total += 1
                    next_replay_allowed_at = (
                        time.monotonic() + PRE_EVENT_DURATION_SEC
                    )
                    next_presence_replay_at = next_replay_allowed_at
                    replay_status = empty_replay_status(
                        temporal_service.threshold,
                        reason="async_running",
                    )

        # 4. EMPTY uses a low-rate person probe. Any fast-motion BURST wakes
        # pose immediately and keeps it hot for the configured hold window.
        now_mono = time.monotonic()
        empty_probe_interval = 1.0 / max(
            0.01, float(edge_policy["empty_probe_hz"])
        )
        idle_probe_due = (
            state_machine.state == AnalysisStateEnum.IDLE
            and now_mono >= next_empty_pose_probe
        )
        should_run_pose = (
            CENTRAL_POSE_ALWAYS_ON
            or burst_active or state_machine.should_run_pose() or idle_probe_due
        )
        # Pace each camera from capture time rather than thread-loop time.
        # Without this gate one camera can submit several frames 2-5 ms apart,
        # consume its quota, and then receive no observation for >150 ms.  The
        # TCN accepts only real 70-150 ms observations, so select them near
        # 10 Hz here instead of manufacturing/interpolating skeleton rows.
        pose_cadence_due = (
            last_pose_request_capture_ts is None
            or packet.capture_mono_ts - last_pose_request_capture_ts
            >= OCCUPIED_POSE_INTERVAL_SEC
        )
        submit_pose = should_run_pose and pose_cadence_due
        pose_res = None
        kpts_xy = None
        kpts_conf = None
        pose_probs = np.zeros(6, dtype=np.float32)
        observation_ts = packet.capture_wall_ts

        if submit_pose:
            last_pose_request_capture_ts = packet.capture_mono_ts
            if now_mono < verify_priority_until:
                request_priority, request_deadline = P0_VERIFY, 0.35
            elif burst_active:
                request_priority, request_deadline = P1_BURST, 0.45
            elif state_machine.state != AnalysisStateEnum.IDLE:
                request_priority, request_deadline = P2_OCCUPIED, 0.80
            else:
                request_priority, request_deadline = P3_EMPTY_PROBE, 1.20
            pose_outcome = scheduler.request_pose(
                camera_id, frame, frame_seq=packet.frame_seq,
                priority=request_priority, deadline_sec=request_deadline,
            )
            pose_res = pose_outcome.result if pose_outcome.completed else None
            if pose_outcome.completed:
                pose_inference_total += 1
                pose_inference_times.append(time.monotonic())
                if len(pose_inference_times) > 120:
                    pose_inference_times.pop(0)
            if state_machine.state == AnalysisStateEnum.IDLE:
                next_empty_pose_probe = now_mono + empty_probe_interval

        # A frame skipped by the 10 Hz pacing gate is not a negative person
        # observation.  Updating the tracker with an empty detection list on
        # those frames made the public status flicker 1 -> 0 -> 1 and also
        # broke otherwise continuous temporal context.  Keep the last
        # published state until an actual inference completes.  This does not
        # manufacture a TCN row: only completed, current-frame observations
        # reach the tracker and temporal adapter below.
        if not submit_pose or pose_res is None:
            continue

        # 5. Extract every person, associate tracks, and select one primary
        # patient. YOLO detection order must never own temporal history.
        in_bed = "NO"
        pose_display = "None"
        pose_conf = 0.0
        person_detected = False
        person_count = 0
        body_in_bed_ratio = 0.0
        feat = {}
        pose_detections: list[PersonDetection] = (
            extract_pose_detections(pose_res[0], bed, fh, fw)
            if pose_res is not None and len(pose_res) > 0 else []
        )

        tracking = person_tracker.update(
            pose_detections,
            observation_ts,
            frame_width=fw,
            frame_height=fh,
        )
        person_count = len(pose_detections)
        for expired_track_id in tracking.expired_track_ids:
            frame_buffers.pop(expired_track_id, None)
            temporal_runners.pop(expired_track_id, None)
        if tracking.primary_switched:
            # A temporal window belongs to one uninterrupted primary identity.
            # Clear every cached runner so switching back cannot resume stale
            # history from a previously selected person.
            tcn_track_reset_total += 1
            temporal_runners.clear()

        kpts_xy = None
        kpts_conf = None
        primary_buffer = None
        if (
            tracking.primary is not None
            and tracking.primary.observed_this_frame
        ):
            kpts_xy = tracking.primary.keypoints_xy
            kpts_conf = tracking.primary.keypoints_conf
            body_in_bed_ratio = float(tracking.primary.bed_overlap_ema)
            feat = enrich_from_keypoints(
                kpts_xy, kpts_conf, bed, motion_state,
                observation_ts, preset, scorer=fall_scorer,
            )
            event_tracker.update(observation_ts, feat)
            person_detected = bool(feat["person_detected"])
            in_bed = "YES" if feat["in_bed"] else "NO"
            primary_buffer = frame_buffers.setdefault(
                tracking.primary.track_id, FrameBuffer(max_frames=30)
            )
            primary_buffer.push(kpts_xy, kpts_conf, timestamp=observation_ts)

        # 6. Update State Machine with person detection
        state_machine.update(
            person_detected,
            bool(primary_buffer and primary_buffer.is_ready(
                state_machine.buffer_ready_threshold
            )),
        )

        # 7. The deployed classifier accepts exactly 34 values (one frame).
        # A trained TCN will consume frame_buffer through a separate adapter.
        should_run_6class = state_machine.should_run_6class()
        if should_run_6class and kpts_xy is not None:
            try:
                pred = keras_clf.predict(kpts_xy.flatten().reshape(1, -1), verbose=0)
                pidx = np.argmax(pred[0])
                pose_probs = np.asarray(pred[0], dtype=np.float32)
                pose_display = CLASS_NAMES[pidx]
                pose_conf = float(pose_probs[pidx])
            except Exception as exc:
                logging.warning(f"6-class prediction error: {exc}")
                pose_display = "ERROR"

        # 7.25 Shadow-only temporal inference owned by primary track ID.
        active_temporal_runner = None
        tcn_track_id = None
        temporal_row_appended = False
        if temporal_service is not None and tracking.primary_track_id is not None:
            tcn_track_id = tracking.primary_track_id
            active_temporal_runner = temporal_runners.setdefault(
                tcn_track_id,
                TemporalShadowRunner(
                    temporal_service,
                    max_interval_sec=LIVE_TCN_MAX_INTERVAL_SEC,
                ),
            )
            if kpts_xy is not None and kpts_conf is not None:
                temporal_status = active_temporal_runner.push(
                    packet.capture_mono_ts, kpts_xy, kpts_conf, pose_probs,
                    timestamp_source="decode_mono_ts",
                )
                temporal_row_appended = (
                    temporal_status.get("last_action") == "append"
                    and temporal_status.get("sample_timestamp") is not None
                    and abs(
                        float(temporal_status["sample_timestamp"])
                        - float(packet.capture_mono_ts)
                    ) < 1e-6
                )
            else:
                active_temporal_runner.observe_gap(packet.capture_mono_ts)
                temporal_status = active_temporal_runner.status()
            if TCN_FUSION_ENABLED and temporal_status.get("candidate"):
                verify_priority_until = max(
                    verify_priority_until, time.monotonic() + 3.0
                )
        else:
            temporal_status = {
                "ready": False, "probability": 0.0, "candidate": False,
                "threshold": (
                    float(temporal_service.threshold)
                    if temporal_service is not None else 0.0
                ),
                "samples": 0, "prediction_count": 0,
                "sample_hz": (
                    float(temporal_service.sample_hz)
                    if temporal_service is not None else 10.0
                ),
                "window_rows": (
                    int(temporal_service.window_rows)
                    if temporal_service is not None else 30
                ),
                "missing_samples_window": 0,
                "missing_samples_total": 0, "sample_timestamp": None,
                "gap_reset_total": 0, "duplicate_skip_total": 0,
                "non_monotonic_skip_total": 0, "last_dt_sec": None,
                "last_action": "not_started", "timestamp_source": "unknown",
                "sampling_contract": "observed_only_70_150ms",
            }

        # A seated or motionless person may be discovered by the low-rate
        # EMPTY probe without opening a new motion episode.  In that case the
        # live 10 Hz runner cannot recover the history that preceded person
        # discovery.  Replay the pre-event ring once per track (and retry at a
        # bounded interval if the first history did not contain 30 contiguous
        # observed poses).  This is independent of the rapid-motion trigger.
        current_track_id = tracking.primary_track_id
        presence_replay_due = (
            current_track_id is not None
            and tracking.primary is not None
            and not bool(temporal_status.get("ready", False))
            and time.monotonic() >= next_presence_replay_at
            and time.monotonic() >= next_replay_allowed_at
        )
        if (
            presence_replay_due
            and PRE_EVENT_REPLAY_ENABLED
            and temporal_service is not None
            and motion_watcher is not None
            and bool(watcher_status.get("pre_event_ready", False))
        ):
            replay_bed = dict(bed or {})
            if replay_bed.get("mask") is not None:
                replay_bed["mask"] = replay_bed["mask"].copy()
            replay_trigger_ts = float(packet.capture_mono_ts)
            submitted = replay_worker.submit(lambda
                replay_bed=replay_bed,
                replay_source_h=fh,
                replay_source_w=fw,
                replay_trigger_ts=replay_trigger_ts:
                run_pre_event_replay(
                    camera_id, motion_watcher, scheduler, keras_replay_clf,
                    temporal_service, replay_bed,
                    source_h=replay_source_h,
                    source_w=replay_source_w,
                    trigger_mono_ts=replay_trigger_ts,
                )
            )
            if submitted:
                replay_attempt_total += 1
                next_replay_allowed_at = (
                    time.monotonic() + PRE_EVENT_DURATION_SEC
                )
                next_presence_replay_at = next_replay_allowed_at
                replay_status = empty_replay_status(
                    temporal_service.threshold,
                    reason="async_running",
                )

        # A valid replay result is a short-lived temporal observation source.
        # Prefer the higher-risk ready source; never inject replay rows into the
        # live runner because their timestamps precede the current frame.
        replay_available = (
            time.monotonic() < replay_valid_until
            and bool(replay_status.get("ready", False))
        )
        effective_temporal_status = temporal_status
        tcn_source = "live" if temporal_status.get("ready") else "none"
        if replay_available and (
            not temporal_status.get("ready")
            or bool(replay_status.get("candidate"))
            or float(replay_status.get("probability", 0.0))
               > float(temporal_status.get("probability", 0.0))
        ):
            effective_temporal_status = replay_status
            tcn_source = "pre_event_replay"

        raw_legacy_fall_score = float(feat.get("fall_score", 0.0))

        # 7.5 Legacy display compatibility. Phase 6 fusion below treats bed
        # position as soft context and does not inherit this hard suppression.
        # 7.5 Safety gate: suppress fall if body mostly stays in bed and pose is bed-safe.
        safe_pose_classes = {"front_lying", "prone_back", "side_near", "side_far", "sitting_center"}
        if person_detected and body_in_bed_ratio >= BODY_IN_BED_SAFE_THRESHOLD and pose_display in safe_pose_classes:
            feat["fall_status"] = "IN_BED_SAFE"
            feat["fall_level"] = "SAFE"
            feat["risk_level"] = "SAFE"
            feat["fall_score"] = 0.0

        # 7.75 Shadow hybrid fusion: temporal + kinematic + posture + bed
        # context. This is observable but cannot trigger production alerts.
        fusion_temporal_candidate = bool(
            TCN_FUSION_ENABLED and effective_temporal_status["candidate"]
        )
        fusion_result = hybrid_fusion.update(FusionInput(
            timestamp=observation_ts,
            track_id=tracking.primary_track_id,
            primary_observed=tracking.primary is not None,
            bed_roi_ready=bool(bed_roi_status["ready"]),
            body_in_bed_ratio=body_in_bed_ratio,
            pose_class=pose_display,
            pose_confidence=pose_conf,
            legacy_fall_score=raw_legacy_fall_score,
            rapid_motion=local_burst_active,
            motion_ratio=float(watcher_status["motion_ratio"]),
            tcn_ready=bool(effective_temporal_status["ready"]),
            tcn_probability=float(effective_temporal_status["probability"]),
            tcn_threshold=float(effective_temporal_status["threshold"]),
            tcn_candidate=fusion_temporal_candidate,
            missing_samples=int(effective_temporal_status.get("missing_samples_window", 0)),
        ))
        if fusion_result.phase.value in {
            "TCN_NOT_READY", "CANDIDATE", "VERIFY", "SHADOW_ALERT"
        }:
            verify_priority_until = max(
                verify_priority_until, time.monotonic() + 3.0
            )

        # Automatic feature-only capture.  This is a dataset trigger, never an
        # alert trigger: predictions only mark a session UNREVIEWED for later
        # human labeling.  Only the exact observed-only 109-D live row is kept.
        if (
            session_recorder is not None
            and camera_id in TEMPORAL_SESSION_CAMERAS
        ):
            current_session_state = {
                # Presence triggers follow the retained primary track, not a
                # single-frame Pose hit. This prevents enter/exit chatter when
                # keypoints are briefly occluded; 109-D rows still require an
                # actual observed primary and are never synthesized.
                "person_observed": tracking.primary_track_id is not None,
                "edge_wake": bool(edge_signal_status.get("wake_active", False)),
                "local_motion": local_burst_active,
                "tcn_candidate": fusion_temporal_candidate,
                "fusion_phase": fusion_result.phase.value,
                "track_id": tracking.primary_track_id,
            }
            session_triggers = derive_temporal_session_triggers(
                previous_session_state, current_session_state
            )
            latest_observation = (
                active_temporal_runner.latest_observation()
                if active_temporal_runner is not None
                and temporal_row_appended
                else None
            )
            if latest_observation is not None and tcn_track_id is not None:
                sample_ts, feature_vector = latest_observation
                pose_quality = float(np.mean(kpts_conf)) if kpts_conf is not None else 0.0
                session_recorder.observe(
                    camera_id,
                    sample_ts,
                    feature_vector,
                    track_id=tcn_track_id,
                    quality=pose_quality,
                    triggers=session_triggers,
                )
            else:
                session_recorder.tick(
                    camera_id,
                    packet.capture_mono_ts,
                    triggers=session_triggers,
                )
            previous_session_state = current_session_state

        # 8. State Update with monitoring
        infer_time = (time.perf_counter() - loop_start) * 1000

        # BURST runs at capture speed. EMPTY remains cheap, and its sleep is
        # interruptible by the watcher below.
        target_fps = (
            20 if CENTRAL_POSE_ALWAYS_ON or burst_active
            else state_machine.get_fps_target()
        )
        # LatestFrameCapture already blocks until a newer source frame exists.
        # Sleeping another source period in central always-on mode skips every
        # second 20 Hz frame (effective 9-12 Hz). Let capture cadence own the
        # loop clock in that mode.
        frame_sleep = (
            0.0 if CENTRAL_POSE_ALWAYS_ON
            else max(0, 1.0 / target_fps - infer_time / 1000.0)
        )
        pose_inference_fps = 0.0
        if len(pose_inference_times) > 1:
            span = pose_inference_times[-1] - pose_inference_times[0]
            if span > 0:
                pose_inference_fps = (len(pose_inference_times) - 1) / span

        scheduler_status = scheduler.metrics(camera_id)
        tracker_status = person_tracker.status()

        with state_lock:
            s = camera_states[camera_id]
            s.in_bed = in_bed
            s.pose = pose_display
            s.pose_conf = pose_conf
            s.fall_score = float(feat.get("fall_score", 0.0))
            s.fall_level = str(feat.get("fall_level", "SAFE"))
            s.fall_status = str(feat.get("fall_status", "NO_PERSON"))
            s.seg_attachment = str(feat.get("seg_attachment", "none"))
            s.risk_level = str(feat.get("risk_level", "SAFE"))
            s.bed_event = event_tracker.bed_event
            # Use UTC ISO format with Z suffix for unambiguous parsing by monitors
            s.timestamp = datetime.utcnow().isoformat() + 'Z'
            s.latency_ms = infer_time
            s.status = "analyzing" if person_detected else "idle"
            s.analysis_state = state_machine.state.value  # NEW: track state machine state
            s.body_in_bed_ratio = body_in_bed_ratio
            s.tcn_shadow_enabled = temporal_service is not None
            s.tcn_shadow_ready = bool(effective_temporal_status['ready'])
            s.tcn_fall_probability = float(effective_temporal_status['probability'])
            s.tcn_alert_candidate = bool(effective_temporal_status['candidate'])
            s.tcn_threshold = float(effective_temporal_status['threshold'])
            s.tcn_samples = int(effective_temporal_status['samples'])
            s.tcn_prediction_count = int(effective_temporal_status['prediction_count'])
            s.tcn_sample_hz = float(effective_temporal_status.get('sample_hz', 10.0))
            s.tcn_window_rows = int(effective_temporal_status.get('window_rows', 30))
            s.tcn_fusion_enabled = TCN_FUSION_ENABLED
            s.tcn_missing_samples_window = int(effective_temporal_status.get('missing_samples_window', 0))
            s.tcn_missing_samples_total = int(effective_temporal_status.get('missing_samples_total', 0))
            s.tcn_last_sample_timestamp = effective_temporal_status.get('sample_timestamp')
            s.tcn_source = tcn_source
            s.tcn_replay_ready = bool(replay_status.get('ready', False))
            s.tcn_replay_probability = float(replay_status.get('probability', 0.0))
            s.tcn_replay_candidate = bool(replay_status.get('candidate', False))
            s.tcn_replay_samples = int(replay_status.get('samples', 0))
            s.tcn_replay_requested_frames = int(replay_status.get('requested_frames', 0))
            s.tcn_replay_observed_frames = int(replay_status.get('observed_pose_frames', 0))
            s.tcn_replay_attempt_total = int(replay_attempt_total)
            s.tcn_replay_completed_total = int(replay_completed_total)
            s.tcn_replay_error_total = int(replay_error_total)
            s.tcn_replay_elapsed_ms = float(replay_status.get('elapsed_ms', 0.0))
            s.tcn_replay_reason = str(replay_status.get('reason', 'not_started'))
            s.tcn_replay_valid_remaining_ms = max(
                0.0, (replay_valid_until - time.monotonic()) * 1000.0
            )
            s.tcn_replay_gap_reset_total = int(
                replay_status.get('gap_reset_total', 0)
            )
            s.analysis_frame_seq = packet.frame_seq
            s.analysis_frame_age_ms = max(
                0.0, (time.monotonic() - packet.capture_mono_ts) * 1000.0
            )
            s.bed_roi_ready = bool(bed_roi_status["ready"])
            s.roi_state = str(bed_roi_status["roi_state"])
            s.bed_roi_restored_from_cache = bool(bed_roi_status["restored_from_cache"])
            s.bed_roi_version = int(bed_roi_status["version"])
            s.bed_roi_source = str(bed_roi_status["source"])
            s.bed_roi_confidence = float(bed_roi_status["confidence"])
            s.bed_roi_agreement_iou = float(bed_roi_status["agreement_iou"])
            s.bed_roi_candidate_count = int(bed_roi_status["candidate_count"])
            s.bed_seg_run_count = int(bed_roi_status["seg_run_count"])
            s.bed_roi_invalid_reason = str(bed_roi_status["invalid_reason"])
            s.runtime_mode = (
                "BURST" if burst_active
                else ("EMPTY" if state_machine.state == AnalysisStateEnum.IDLE else "OCCUPIED")
            )
            s.watcher_fps = (
                float(watcher_status["watcher_fps"])
                if watcher_status["watcher_thread_alive"] else 0.0
            )
            s.watcher_frame_seq = int(watcher_status["watcher_frame_seq"])
            s.watcher_processed_total = int(watcher_status["watcher_processed_total"])
            s.watcher_thread_alive = bool(watcher_status["watcher_thread_alive"])
            s.motion_ratio = float(watcher_status["motion_ratio"])
            s.motion_detected = bool(watcher_status["motion_detected"])
            s.motion_hit_streak = int(watcher_status["motion_hit_streak"])
            s.motion_trigger_total = int(watcher_status["motion_trigger_total"])
            s.burst_active = burst_active
            s.burst_remaining_ms = float(watcher_status["burst_remaining_ms"])
            s.edge_signal_connected = bool(edge_signal_status["connected"])
            s.edge_signal_wake_active = edge_wake_active
            s.edge_signal_result_fresh = bool(edge_signal_status["result_fresh"])
            s.edge_signal_result_age_ms = edge_signal_status["result_age_ms"]
            s.edge_signal_person_present = bool(edge_signal_status["person_present"])
            s.edge_signal_last_person_present = bool(
                edge_signal_status.get("last_person_present", False)
            )
            s.edge_signal_pose_confidence = float(edge_signal_status["pose_confidence"])
            s.edge_signal_quality = float(edge_signal_status["quality"])
            s.edge_signal_frame_seq = edge_signal_status["frame_seq"]
            s.edge_signal_model_bundle_version = edge_signal_status["model_bundle_version"]
            s.edge_signal_error_total = int(edge_signal_status["error_total"])
            s.edge_managed = edge_managed
            s.edge_local_watcher_suppressed = bool(edge_policy["suppress_local_watcher"])
            s.edge_fallback_active = bool(edge_policy["fallback_active"])
            s.edge_effective_empty_probe_hz = float(edge_policy["empty_probe_hz"])
            s.pre_event_frames = int(watcher_status["pre_event_frames"])
            s.pre_event_coverage_sec = float(watcher_status["pre_event_coverage_sec"])
            s.pre_event_bytes = int(watcher_status["pre_event_bytes"])
            s.pre_event_target_sec = float(watcher_status["pre_event_target_sec"])
            s.pre_event_sample_hz = float(watcher_status["pre_event_sample_hz"])
            s.pre_event_ready = bool(watcher_status["pre_event_ready"])
            s.pose_inference_total = int(pose_inference_total)
            s.pose_inference_fps = float(pose_inference_fps)
            s.scheduler_completed_total = int(scheduler_status['completed_total'])
            s.scheduler_completed_hz = float(scheduler_status['completed_hz'])
            s.scheduler_queue_latency_ms = float(scheduler_status['last_queue_latency_ms'])
            s.scheduler_inference_ms = float(scheduler_status['last_inference_ms'])
            s.scheduler_stale_drop_total = int(scheduler_status['stale_drop_total'])
            s.scheduler_superseded_drop_total = int(scheduler_status['superseded_drop_total'])
            s.scheduler_timeout_total = int(scheduler_status['timeout_total'])
            s.scheduler_error_total = int(scheduler_status['error_total'])
            s.scheduler_pending = int(scheduler_status['pending'])
            s.scheduler_last_priority = scheduler_status['last_priority']
            s.scheduler_last_model = scheduler_status['last_model']
            s.scheduler_thread_alive = bool(scheduler_status['thread_alive'])
            s.person_count = int(person_count)
            s.track_count = int(tracker_status['track_count'])
            s.primary_track_id = tracker_status['primary_track_id']
            s.track_switch_total = int(tracker_status['track_switch_total'])
            s.track_created_total = int(tracker_status['track_created_total'])
            s.track_expired_total = int(tracker_status['track_expired_total'])
            s.primary_track_observed = bool(tracker_status['primary_observed'])
            s.primary_track_bed_overlap = float(tracker_status['primary_bed_overlap_ema'])
            s.primary_track_confidence = float(tracker_status['primary_confidence_ema'])
            s.tcn_track_id = tcn_track_id
            s.tcn_track_reset_total = int(tcn_track_reset_total)
            # This public counter is live-only.  Mixing the selected replay
            # source here made the counter appear to decrease when source
            # selection changed and hid actual live cadence health.
            s.tcn_gap_reset_total = int(temporal_status.get('gap_reset_total', 0))
            s.tcn_duplicate_skip_total = int(
                effective_temporal_status.get('duplicate_skip_total', 0)
            )
            s.tcn_non_monotonic_skip_total = int(
                effective_temporal_status.get('non_monotonic_skip_total', 0)
            )
            s.tcn_last_dt_sec = effective_temporal_status.get('last_dt_sec')
            s.tcn_last_action = str(
                effective_temporal_status.get('last_action', 'not_started')
            )
            s.fusion_phase = fusion_result.phase.value
            s.fusion_risk = float(fusion_result.risk)
            s.fusion_evidence = list(fusion_result.evidence)
            s.fusion_safe_evidence = list(fusion_result.safe_evidence)
            s.fusion_candidate_age_sec = float(fusion_result.candidate_age_sec)
            s.fusion_quality = float(fusion_result.quality)
            s.fusion_track_id = fusion_result.track_id
            recorder_status = (
                shadow_recorder.status() if shadow_recorder is not None
                else {"enabled": False, "thread_alive": False, "written_total": 0,
                      "dropped_total": 0, "error_total": 0, "queue_depth": 0}
            )
            s.feature_recorder_enabled = bool(recorder_status["enabled"])
            s.feature_recorder_thread_alive = bool(recorder_status["thread_alive"])
            s.feature_recorder_written_total = int(recorder_status["written_total"])
            s.feature_recorder_dropped_total = int(recorder_status["dropped_total"])
            s.feature_recorder_error_total = int(recorder_status["error_total"])
            s.feature_recorder_queue_depth = int(recorder_status["queue_depth"])

            frame_times.append(time.time())
            if len(frame_times) > 30: frame_times.pop(0)
            if len(frame_times) > 1:
                s.pipeline_fps = len(frame_times) / (frame_times[-1] - frame_times[0])
            recorder_snapshot = s.model_dump()

        if shadow_recorder is not None:
            capture_metrics = capture.metrics()
            recorder_snapshot.update({
                "capture_connected": bool(capture_metrics["connected"]),
                "capture_fps": float(capture_metrics["capture_fps"]),
                "capture_decode_error_total": int(capture_metrics["decode_error_total"]),
            })
            shadow_recorder.submit(
                camera_id, recorder_snapshot, mono_ts=packet.capture_mono_ts
            )

        # EMPTY sleep is interruptible: watcher motion wakes inference now.
        if frame_sleep > 0:
            if motion_watcher is not None and motion_watcher.is_alive():
                motion_watcher.wait_for_burst(frame_sleep)
            else:
                time.sleep(frame_sleep)

@app.get("/recorder/status")
def get_recorder_status():
    if shadow_recorder is None:
        return {"enabled": False, "thread_alive": False}
    return shadow_recorder.status()


@app.get("/temporal-recorder/status")
def get_temporal_recorder_status():
    if temporal_session_recorder is None:
        return {"enabled": False, "thread_alive": False}
    status = temporal_session_recorder.status()
    status["cameras"] = sorted(TEMPORAL_SESSION_CAMERAS)
    return status

@app.get("/calibration/status")
def get_calibration_status():
    if not CALIBRATION_REPORT_PATH.exists():
        return {"available": False, "readiness": "NOT_READY"}
    try:
        report = json.loads(CALIBRATION_REPORT_PATH.read_text(encoding="utf-8"))
        return {
            "available": True,
            "generated_at": report.get("generated_at"),
            "readiness": report.get("readiness", "NOT_READY"),
            "targets": report.get("targets", {}),
            "overall": report.get("overall", {}),
        }
    except Exception as exc:
        return {
            "available": False,
            "readiness": "NOT_READY",
            "error": f"{type(exc).__name__}: {exc}",
        }

@app.get("/status")
def get_status():
    with state_lock:
        result = {cid: s.model_dump() for cid, s in camera_states.items()}
    for cid, item in result.items():
        capture = camera_captures.get(cid)
        if capture is None:
            continue
        metrics = capture.metrics()
        item["capture_connected"] = bool(metrics["connected"])
        item["capture_fps"] = float(metrics["capture_fps"])
        item["capture_frame_seq"] = int(metrics["frame_seq"])
        item["capture_frame_age_ms"] = metrics["frame_age_ms"]
        item["capture_decode_error_total"] = int(metrics["decode_error_total"])
        item["capture_reconnect_total"] = int(metrics["reconnect_total"])
        manager = bed_roi_managers.get(cid)
        if manager is not None:
            roi = manager.status()
            item["bed_roi_ready"] = bool(roi["ready"])
            item["roi_state"] = str(roi["roi_state"])
            item["bed_roi_restored_from_cache"] = bool(roi["restored_from_cache"])
            item["bed_roi_version"] = int(roi["version"])
            item["bed_roi_source"] = str(roi["source"])
            item["bed_roi_confidence"] = float(roi["confidence"])
            item["bed_roi_agreement_iou"] = float(roi["agreement_iou"])
            item["bed_roi_candidate_count"] = int(roi["candidate_count"])
            item["bed_seg_run_count"] = int(roi["seg_run_count"])
            item["bed_roi_invalid_reason"] = str(roi["invalid_reason"])
        if edge_signal_client is not None:
            edge = edge_signal_client.status(cid)
            item["edge_signal_connected"] = bool(edge["connected"])
            item["edge_signal_wake_active"] = bool(edge["wake_active"])
            item["edge_signal_result_fresh"] = bool(edge["result_fresh"])
            item["edge_signal_result_age_ms"] = edge["result_age_ms"]
            item["edge_signal_person_present"] = bool(edge["person_present"])
            item["edge_signal_last_person_present"] = bool(
                edge.get("last_person_present", False)
            )
            item["edge_signal_pose_confidence"] = float(edge["pose_confidence"])
            item["edge_signal_quality"] = float(edge["quality"])
            item["edge_signal_frame_seq"] = edge["frame_seq"]
            item["edge_signal_model_bundle_version"] = edge["model_bundle_version"]
            item["edge_signal_error_total"] = int(edge["error_total"])
        watcher = motion_watchers.get(cid)
        if watcher is not None:
            motion = watcher.status()
            edge_managed = cid in EDGE_MANAGED_CAMERAS
            edge_connected = bool(item.get("edge_signal_connected", False))
            edge_healthy = bool(
                edge_connected and item.get("edge_signal_result_fresh", False)
            )
            item["edge_managed"] = edge_managed
            item["edge_local_watcher_suppressed"] = bool(
                edge_managed and edge_healthy and not motion["watcher_thread_alive"]
            )
            item["edge_fallback_active"] = bool(
                edge_managed and not edge_healthy and motion["watcher_thread_alive"]
            )
            item["edge_effective_empty_probe_hz"] = float(
                EDGE_MANAGED_EMPTY_PROBE_HZ
                if item["edge_local_watcher_suppressed"]
                else EMPTY_POSE_PROBE_HZ
            )
            item["watcher_fps"] = (
                float(motion["watcher_fps"])
                if motion["watcher_thread_alive"] else 0.0
            )
            item["watcher_frame_seq"] = int(motion["watcher_frame_seq"])
            item["watcher_processed_total"] = int(motion["watcher_processed_total"])
            item["watcher_thread_alive"] = bool(motion["watcher_thread_alive"])
            item["motion_ratio"] = float(motion["motion_ratio"])
            item["motion_detected"] = bool(motion["motion_detected"])
            item["motion_hit_streak"] = int(motion["motion_hit_streak"])
            item["motion_trigger_total"] = int(motion["motion_trigger_total"])
            item["burst_active"] = (
                bool(motion["burst_active"])
                or bool(item.get("edge_signal_wake_active", False))
            )
            item["burst_remaining_ms"] = float(motion["burst_remaining_ms"])
            item["pre_event_frames"] = int(motion["pre_event_frames"])
            item["pre_event_coverage_sec"] = float(motion["pre_event_coverage_sec"])
            item["pre_event_bytes"] = int(motion["pre_event_bytes"])
            item["pre_event_target_sec"] = float(motion["pre_event_target_sec"])
            item["pre_event_sample_hz"] = float(motion["pre_event_sample_hz"])
            item["pre_event_ready"] = bool(motion["pre_event_ready"])
            item["runtime_mode"] = (
                "BURST" if item["burst_active"]
                else ("EMPTY" if item["analysis_state"] == "idle" else "OCCUPIED")
            )
        if inference_scheduler is not None:
            sched = inference_scheduler.metrics(cid)
            item["scheduler_completed_total"] = int(sched["completed_total"])
            item["scheduler_completed_hz"] = float(sched["completed_hz"])
            item["scheduler_queue_latency_ms"] = float(sched["last_queue_latency_ms"])
            item["scheduler_inference_ms"] = float(sched["last_inference_ms"])
            item["scheduler_stale_drop_total"] = int(sched["stale_drop_total"])
            item["scheduler_superseded_drop_total"] = int(sched["superseded_drop_total"])
            item["scheduler_timeout_total"] = int(sched["timeout_total"])
            item["scheduler_error_total"] = int(sched["error_total"])
            item["scheduler_pending"] = int(sched["pending"])
            item["scheduler_last_priority"] = sched["last_priority"]
            item["scheduler_last_model"] = sched["last_model"]
            item["scheduler_thread_alive"] = bool(sched["thread_alive"])
    return result


def _readiness_snapshot() -> dict:
    scheduler_alive = False
    if inference_scheduler is not None:
        first_camera = next(iter(CAMERA_CONFIGS))
        scheduler_alive = bool(
            inference_scheduler.metrics(first_camera)["thread_alive"]
        )
    capture_threads_alive = (
        len(camera_captures) == len(CAMERA_CONFIGS)
        and all(capture.is_alive() for capture in camera_captures.values())
    )
    watcher_threads_alive = (
        len(motion_watchers) == len(CAMERA_CONFIGS)
        and all(
            (
                watcher.status()["watcher_thread_alive"]
                or (
                    cid in EDGE_MANAGED_CAMERAS
                    and edge_signal_client is not None
                    and edge_signal_client.status(cid)["connected"]
                )
            )
            for cid, watcher in motion_watchers.items()
        )
    )
    recorder_status = get_recorder_status()
    recorder_ready = (
        not SHADOW_RECORD_ENABLED
        or bool(recorder_status.get("thread_alive", False))
    )
    temporal_recorder_status = get_temporal_recorder_status()
    temporal_recorder_ready = (
        not TEMPORAL_SESSION_RECORD_ENABLED
        or bool(temporal_recorder_status.get("thread_alive", False))
    )
    checks = {
        "analysis_running": bool(analysis_running),
        "inference_models_loaded": inference_pool is not None,
        "scheduler_thread_alive": scheduler_alive,
        "capture_threads_alive": capture_threads_alive,
        "watcher_threads_alive": watcher_threads_alive,
        "feature_recorder_ready": recorder_ready,
        "temporal_session_recorder_ready": temporal_recorder_ready,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "uptime_sec": max(0.0, time.monotonic() - PROCESS_STARTED_MONO),
    }


@app.get("/health/live")
def health_live():
    return {
        "live": True,
        "uptime_sec": max(0.0, time.monotonic() - PROCESS_STARTED_MONO),
    }


@app.get("/health/ready")
def health_ready():
    snapshot = _readiness_snapshot()
    return JSONResponse(snapshot, status_code=200 if snapshot["ready"] else 503)


@app.get("/health/cameras")
def health_cameras():
    return evaluate_fleet(get_status())


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    readiness = _readiness_snapshot()
    return render_prometheus(
        get_status(),
        process_ready=bool(readiness["ready"]),
        recorder=get_recorder_status(),
    )


@app.get("/api/v2/status")
def api_v2_status():
    return get_status()


@app.get("/api/v2/status/{camera_id}")
def api_v2_camera_status(camera_id: str):
    state = get_status().get(camera_id)
    if state is None:
        return JSONResponse(
            {"error": "unknown_camera", "camera_id": camera_id},
            status_code=404,
        )
    return state


@app.get("/viewer", response_class=HTMLResponse)
def viewer():

    camera_html = ""
    for cid, cfg in CAMERA_CONFIGS.items():
        camera_html += f"""
        <div class="camera-card">
            <h3>{cfg['name']}</h3>
            <div class="video-wrap"><img src="/video/{cid}" id="img-{cid}" alt="{cfg['name']}" style="width:100%;height:auto;"/></div>
            <div class="status-row"><span class="label">In Bed:</span> <span class="value" id="inbed-{cid}">--</span></div>
            <div class="status-row"><span class="label">Pose:</span> <span class="value" id="pose-{cid}">--</span></div>
            <div class="status-row"><span class="label">Fall Score:</span> <span class="value" id="score-{cid}">--</span></div>
            <div class="status-row"><span class="label">Risk:</span> <span class="value" id="risk-{cid}">--</span></div>
            <div class="status-row"><span class="label">Body in Bed:</span> <span class="value" id="bodyinbed-{cid}">--</span></div>
            <div class="status-row"><span class="label">Event:</span> <span class="value" id="event-{cid}">--</span></div>
            <div class="status-row"><span class="label">Status:</span> <span class="value" id="status-{cid}">--</span></div>
            <div class="status-row"><span class="label">Analysis State:</span> <span class="value" id="analysis-{cid}" style="color: #ffaa00;">--</span></div>
            <div class="status-row"><span class="label">Runtime Mode:</span> <span class="value" id="runtime-{cid}">--</span></div>
            <div class="status-row"><span class="label">Motion Watcher:</span> <span class="value" id="motion-{cid}">--</span></div>
            <div class="status-row"><span class="label">Pre-event Ring:</span> <span class="value" id="preroll-{cid}">--</span></div>
            <div class="status-row"><span class="label">Pose Load:</span> <span class="value" id="poseload-{cid}">--</span></div>
            <div class="status-row"><span class="label">Primary Track:</span> <span class="value" id="track-{cid}">--</span></div>
            <div class="status-row"><span class="label">GPU Scheduler:</span> <span class="value" id="scheduler-{cid}">--</span></div>
            <div class="status-row"><span class="label">Capture:</span> <span class="value" id="capture-{cid}">--</span></div>
            <div class="status-row"><span class="label">Auto Bed ROI:</span> <span class="value" id="bedroi-{cid}">--</span></div>
            <div class="status-row"><span class="label">TCN Shadow:</span> <span class="value" id="tcn-{cid}">--</span></div>
            <div class="status-row"><span class="label">Hybrid Shadow:</span> <span class="value" id="fusion-{cid}">--</span></div>
            <div class="status-row"><span class="label">Feature Recorder:</span> <span class="value" id="recorder-{cid}">--</span></div>
        </div>
        """
    return f"""
    <html>
    <head>
        <title>Multi-Camera Monitoring (Unified)</title>
        <style>
            body {{ font-family: sans-serif; background: #1a1a2e; color: #fff; padding: 20px; }}
            .container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }}
            .camera-card {{ background: #16213e; padding: 15px; border-radius: 10px; border-left: 5px solid #0f3460; }}
            .status-row {{ display: flex; justify-content: space-between; margin: 5px 0; border-bottom: 1px solid #1f4068; }}
            .label {{ color: #999; }}
            .value {{ font-weight: bold; color: #4ecca3; }}
            .video-wrap {{ width: 100%; height: 220px; overflow: hidden; display:flex; align-items:center; justify-content:center; background:#0b1226; margin-bottom:8px; }}
            .video-wrap img {{ width: 100%; height: auto; max-height: 100%; object-fit: cover; }}
        </style>
        <script>
            let statusUpdateInFlight = false;
            async function update() {{
                if (statusUpdateInFlight) return;
                statusUpdateInFlight = true;
                try {{
                    const res = await fetch('/status', {{ cache: 'no-store' }});
                    const data = await res.json();
                    for (const [id, s] of Object.entries(data)) {{
                    document.getElementById(`inbed-${{id}}`).textContent = s.in_bed;
                    document.getElementById(`pose-${{id}}`).textContent = s.pose;
                    document.getElementById(`score-${{id}}`).textContent = s.fall_score.toFixed(0);
                    document.getElementById(`risk-${{id}}`).textContent = s.risk_level;
                    document.getElementById(`bodyinbed-${{id}}`).textContent = `${{((s.body_in_bed_ratio || 0) * 100).toFixed(0)}}%`;
                    document.getElementById(`event-${{id}}`).textContent = s.bed_event || '-';
                    document.getElementById(`status-${{id}}`).textContent = s.status;
                    document.getElementById(`analysis-${{id}}`).textContent = s.analysis_state;
                    document.getElementById(`runtime-${{id}}`).textContent = s.runtime_mode;
                    document.getElementById(`motion-${{id}}`).textContent =
                        `${{(s.watcher_fps || 0).toFixed(1)}} FPS / motion ${{((s.motion_ratio || 0) * 100).toFixed(1)}}% / triggers ${{s.motion_trigger_total || 0}}`;
                    document.getElementById(`preroll-${{id}}`).textContent =
                        `${{s.pre_event_ready ? 'READY' : 'FILLING'}} / ${{(s.pre_event_coverage_sec || 0).toFixed(1)}}s / ${{s.pre_event_frames || 0}} frames / ${{((s.pre_event_bytes || 0) / 1048576).toFixed(1)}} MiB`;
                    document.getElementById(`poseload-${{id}}`).textContent =
                        `${{(s.pose_inference_fps || 0).toFixed(2)}} FPS / total ${{s.pose_inference_total || 0}}`;
                    document.getElementById(`track-${{id}}`).textContent =
                        `ID ${{s.primary_track_id ?? '-'}} / persons ${{s.person_count || 0}} / tracks ${{s.track_count || 0}} / switches ${{s.track_switch_total || 0}}`;
                    const schedDrops = (s.scheduler_stale_drop_total || 0) +
                        (s.scheduler_superseded_drop_total || 0) + (s.scheduler_timeout_total || 0);
                    document.getElementById(`scheduler-${{id}}`).textContent =
                        `${{(s.scheduler_completed_hz || 0).toFixed(2)}} Hz / q ${{(s.scheduler_queue_latency_ms || 0).toFixed(0)}}ms / P${{s.scheduler_last_priority ?? '-'}} / drops ${{schedDrops}}`;
                    const captureAge = s.capture_frame_age_ms == null ? '--' : `${{s.capture_frame_age_ms.toFixed(0)}}ms`;
                    document.getElementById(`capture-${{id}}`).textContent = s.capture_connected
                        ? `${{s.capture_fps.toFixed(1)}} FPS / ${{captureAge}}`
                        : 'disconnected';
                    const roiText = s.bed_roi_ready
                        ? `READY v${{s.bed_roi_version}} / IoU ${{(s.bed_roi_agreement_iou || 0).toFixed(2)}} / seg ${{s.bed_seg_run_count}}`
                        : `ROI_NOT_READY / ${{s.bed_roi_invalid_reason || 'detecting'}} / ${{s.bed_roi_candidate_count || 0}}`;
                    document.getElementById(`bedroi-${{id}}`).textContent = roiText;
                    const tcnText = s.tcn_shadow_enabled
                        ? `${{s.tcn_shadow_ready ? (s.tcn_fall_probability * 100).toFixed(1) + '%' : 'warming ' + s.tcn_samples + '/' + (s.tcn_window_rows || 30)}} / ${{s.tcn_sample_hz || 0}}Hz / source ${{s.tcn_source || 'none'}} / fusion ${{s.tcn_fusion_enabled ? 'on' : 'telemetry-only'}} / replay ${{s.tcn_replay_reason || '-'}} ${{s.tcn_replay_observed_frames || 0}}/${{s.tcn_replay_requested_frames || 0}} ${{(s.tcn_replay_elapsed_ms || 0).toFixed(0)}}ms / owner ${{s.tcn_track_id ?? '-'}}${{s.tcn_alert_candidate ? ' RAW-CANDIDATE' : ''}}`
                        : 'disabled';
                    document.getElementById(`tcn-${{id}}`).textContent = tcnText;
                    const fusionEvidence = (s.fusion_evidence || []).join(',') || '-';
                    document.getElementById(`fusion-${{id}}`).textContent =
                        `${{s.fusion_phase}} / ${{((s.fusion_risk || 0) * 100).toFixed(1)}}% / owner ${{s.fusion_track_id ?? '-'}} / ${{fusionEvidence}}`;
                    document.getElementById(`recorder-${{id}}`).textContent =
                        s.feature_recorder_enabled
                            ? `${{s.feature_recorder_thread_alive ? 'ON' : 'STOPPED'}} / rows ${{s.feature_recorder_written_total || 0}} / q ${{s.feature_recorder_queue_depth || 0}} / drop ${{s.feature_recorder_dropped_total || 0}}`
                            : 'disabled';
                    }}
                }} finally {{
                    statusUpdateInFlight = false;
                }}
            }}
            window.addEventListener('DOMContentLoaded', () => {{
                update();
                setInterval(update, 100);
            }});
        </script>
    </head>
    <body>
        <h1>🛏️ Multi-Camera Logic Unified</h1>
        <div class="container">{camera_html}</div>
    </body>
    </html>
    """

@app.get('/image/{camera_id}')
def get_latest_image(camera_id: str):
    """Return latest frame as JPEG for polling-based display."""
    capture = camera_captures.get(camera_id)
    packet = capture.latest() if capture is not None else None
    if packet is None:
        # Return blank placeholder (dark image)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buf = cv2.imencode('.jpg', blank)
        jpg = buf.tobytes() if ret else b''
    else:
        ret, buf = cv2.imencode('.jpg', packet.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg = buf.tobytes() if ret else b''

    return StreamingResponse(iter([jpg]), media_type='image/jpeg')

@app.get('/video/{camera_id}')
def video_stream(camera_id: str):
    """Return latest-frame MJPEG independently from the inference loop."""
    capture = camera_captures.get(camera_id)

    def gen():
        last_version = -1
        while True:
            if capture is None:
                time.sleep(0.25)
                continue
            packet = capture.wait_for_frame(last_version, timeout=1.0)
            if packet is None:
                continue
            try:
                ret, buf = cv2.imencode('.jpg', packet.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    time.sleep(0.02)
                    continue
                last_version = packet.frame_seq
                jpg = buf.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.05)
                continue

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("POSE_SERVER_PORT", "8000")))
