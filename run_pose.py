import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
import tensorflow as tf
import keras
import logging
import imutils
import time
import math
from collections import deque
from pathlib import Path
from ultralytics import YOLO

from bed_roi.roi_utils import apply_bed_roi, load_bed_roi
from server import (
    BED_SEG_CONF,
    draw_bed_overlay,
    draw_bed_zones,
    extract_bed_detection,
    is_person_in_bed,
)

tf.config.set_visible_devices([], 'GPU')

logging.basicConfig(level=logging.INFO, format='%(message)s')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_SEG_CLASS = 0
BED_ROI_PATH = os.environ.get('POSE_BED_ROI', os.path.join(BASE_DIR, 'bed_roi/bed_roi.json'))
USE_BED_ROI = os.environ.get('POSE_USE_BED_ROI', '1') != '0'
YOLO_DEVICE = os.environ.get('POSE_YOLO_DEVICE', '0')

# -- 난간 키포인트 인덱스 정의 --
RAIL_KEYPOINTS = {
    'Rail_0': 0,
    'Rail_1': 1,
}
RAIL_COLORS = {
    'Rail_0': (0, 255, 0),  # 초록
    'Rail_1': (0, 255, 0),
}

# -- 낙상 예방용 키포인트 인덱스 정의 --
FALL_KEYPOINTS = {
    'L_elbow':  7,
    'R_elbow':  8,
    'L_hip':   11,
    'R_hip':   12,
    'L_knee':  13,
    'R_knee':  14,
}
KEYPOINT_COLORS = {
    'L_elbow': (0,   255, 255),  # 노랑
    'R_elbow': (0,   255, 255),
    'L_hip':   (255,   0, 255),  # 보라
    'R_hip':   (255,   0, 255),
    'L_knee':  (0,   165, 255),  # 주황
    'R_knee':  (0,   165, 255),
}


class FallDetector:
    def __init__(self):
        self.history = deque(maxlen=90)
        self.state = "NORMAL"
        self.candidate_since = None
        self.confirmed_since = None
        self.cooldown_until = 0.0

    @staticmethod
    def _center(p1, p2):
        if p1 is None or p2 is None:
            return None
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    @staticmethod
    def _to_point(arr, idx):
        if arr.shape[0] <= idx:
            return None
        x, y = arr[idx]
        if x <= 1 and y <= 1:
            return None
        return (float(x), float(y))

    def _torso_angle_deg(self, left_shoulder, right_shoulder, left_hip, right_hip):
        shoulder_c = self._center(left_shoulder, right_shoulder)
        hip_c = self._center(left_hip, right_hip)
        if shoulder_c is None or hip_c is None:
            return None
        dx = shoulder_c[0] - hip_c[0]
        dy = shoulder_c[1] - hip_c[1]
        return abs(math.degrees(math.atan2(dy, dx)))

    def update(self, keypoints_xy, person_bbox, in_bed_status):
        now = time.time()
        if keypoints_xy is None or person_bbox is None:
            return self.state, 0.0, False

        l_sh = self._to_point(keypoints_xy, 5)
        r_sh = self._to_point(keypoints_xy, 6)
        l_hip = self._to_point(keypoints_xy, 11)
        r_hip = self._to_point(keypoints_xy, 12)

        hip_c = self._center(l_hip, r_hip)
        shoulder_c = self._center(l_sh, r_sh)

        if hip_c is None:
            return self.state, 0.0, False

        x1, y1, x2, y2 = person_bbox[:4]
        bbox_h = max(float(y2 - y1), 1.0)
        torso_angle = self._torso_angle_deg(l_sh, r_sh, l_hip, r_hip)

        motion = 0.0
        if self.history:
            prev = self.history[-1]
            dt = max(now - prev["t"], 1e-3)
            motion = abs(hip_c[1] - prev["hip_y"]) / dt
            v_speed = (hip_c[1] - prev["hip_y"]) / dt
        else:
            v_speed = 0.0

        # Scale by person size so thresholds are less camera-dependent.
        norm_v = v_speed / bbox_h
        norm_motion = motion / bbox_h

        self.history.append({
            "t": now,
            "hip_y": hip_c[1],
            "shoulder_y": shoulder_c[1] if shoulder_c else None,
            "bbox_h": bbox_h,
            "torso_angle": torso_angle,
            "norm_v": norm_v,
            "norm_motion": norm_motion,
        })

        rapid_drop = norm_v > 0.75
        horizontal_body = torso_angle is not None and torso_angle < 30.0
        low_motion = norm_motion < 0.15

        if now < self.cooldown_until:
            self.state = "COOLDOWN"
        elif self.state == "NORMAL":
            if rapid_drop and (horizontal_body or in_bed_status == "NO"):
                self.state = "CANDIDATE"
                self.candidate_since = now
        elif self.state == "CANDIDATE":
            if rapid_drop or horizontal_body:
                if self.candidate_since and (now - self.candidate_since) > 0.35 and low_motion:
                    self.state = "CONFIRMED"
                    self.confirmed_since = now
            elif self.candidate_since and (now - self.candidate_since) > 1.0:
                self.state = "NORMAL"
                self.candidate_since = None
        elif self.state == "CONFIRMED":
            if self.confirmed_since and (now - self.confirmed_since) > 2.0:
                self.state = "COOLDOWN"
                self.cooldown_until = now + 3.0
        elif self.state == "COOLDOWN":
            if now >= self.cooldown_until:
                self.state = "NORMAL"
                self.candidate_since = None
                self.confirmed_since = None

        score = 0.0
        score += min(max(norm_v, 0.0), 1.5) * 0.5
        score += (0.35 if horizontal_body else 0.0)
        score += (0.15 if low_motion else 0.0)
        score = min(score, 1.0)
        alert = self.state == "CONFIRMED"
        return self.state, score, alert


class MyConfig:
    def __init__(self):
        self.yolo_seg_weight  = self._resolve_path('yolo11n-bed-seg.pt')
        self.yolo_pose_weight = self._resolve_path('yolo11m-pose.pt')
        self.pose_keras_model = self._resolve_path('my_model_six_check.keras')
        self.rtsp_url         = os.environ.get(
            'POSE_RTSP_URL', 'rtsp://192.168.0.161:8554/stream'
        )

    @staticmethod
    def _resolve_path(path_value):
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(BASE_DIR, path_value)


def extract_keypoints(keypoints_xy, kpts_conf, kpt_dict, conf_threshold=0.3):
    if keypoints_xy.shape[0] < 17:
        return {}
    result = {}
    for name, idx in kpt_dict.items():
        x, y = keypoints_xy[idx]
        if kpts_conf is not None and kpts_conf[idx] < conf_threshold:
            result[name] = None
            continue
        if x < 1 and y < 1:
            result[name] = None
        else:
            result[name] = (int(x), int(y))
    return result


def draw_keypoints(frame, kpt_dict, color_dict):
    for name, pt in kpt_dict.items():
        if pt is None:
            continue
        color = color_dict[name]
        cv2.circle(frame, pt, 8, color, -1)
        cv2.circle(frame, pt, 10, (255, 255, 255), 2)
        cv2.putText(frame, name, (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame


def main(params):
    if not os.path.exists(params.pose_keras_model):
        logging.error(f"모델 파일 없음: {params.pose_keras_model}")
        return

    try:
        yolo_seg_model  = YOLO(params.yolo_seg_weight)
        yolo_pose_model = YOLO(params.yolo_pose_weight)
        keras_clf       = keras.models.load_model(params.pose_keras_model)
    except Exception as e:
        logging.error(f"로딩 에러: {e}")
        return

    class_names = [
        "정면_누움",
        "엎드림_등",
        "옆누움_가까움",
        "옆누움_멀음",
        "앉음_중앙",
        "앉음_가장자리",
    ]

    cap = cv2.VideoCapture(params.rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fall_detector = FallDetector()
    use_gui = bool(os.environ.get("DISPLAY"))
    if not use_gui:
        logging.warning("[WARN] DISPLAY 없음: cv2.imshow 비활성화(헤드리스 모드)")

    cached_bed = None
    frame_idx = 0
    seg_every = max(1, int(os.environ.get('POSE_SEG_EVERY', '3')))

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = imutils.resize(frame, width=640)
        fh, fw = frame.shape[:2]
        frame_idx += 1

        if frame_idx % seg_every == 1 or cached_bed is None:
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

        pose_res = yolo_pose_model.predict(
            frame, conf=0.5, device=YOLO_DEVICE, verbose=False
        )

        in_bed_status = "NO"
        in_bed_method = "none"
        pose_label    = "None"
        rail_kpts     = {}
        fall_kpts     = {}

        display_frame = draw_bed_overlay(frame, bed, roi_bbox)

        if len(pose_res) > 0 and len(pose_res[0].keypoints) > 0:
            person_bbox = pose_res[0].boxes.xyxy[0].cpu().numpy()
            in_bed_yes, in_bed_method = is_person_in_bed(person_bbox, bed, fh, fw)
            if in_bed_yes:
                in_bed_status = "YES"

            display_frame = pose_res[0].plot(img=display_frame)
            display_frame = draw_bed_zones(display_frame, bed.get('bbox'))

            # D. 키포인트 추출
            kpts_xy   = pose_res[0].keypoints[0].xy.cpu().numpy()
            kpts_conf = pose_res[0].keypoints[0].conf.cpu().numpy()

            rail_kpts = extract_keypoints(kpts_xy, kpts_conf, RAIL_KEYPOINTS)
            fall_kpts = extract_keypoints(kpts_xy, kpts_conf, FALL_KEYPOINTS)

            # E. Keras 자세 분류
            kpts_data = kpts_xy.flatten()
            if len(kpts_data) == 34:
                kpts_input = kpts_data.reshape(1, -1).astype('float32')
                pred       = keras_clf.predict(kpts_input, verbose=0)
                pose_label = class_names[np.argmax(pred)]
                fall_state, fall_score, fall_alert = fall_detector.update(kpts_xy, person_bbox, in_bed_status)
            else:
                fall_state, fall_score, fall_alert = "NORMAL", 0.0, False
        else:
            fall_state, fall_score, fall_alert = "NORMAL", 0.0, False

        # F. 난간 + 낙상 키포인트 오버레이
        if rail_kpts:
            display_frame = draw_keypoints(display_frame, rail_kpts, RAIL_COLORS)
        if fall_kpts:
            display_frame = draw_keypoints(display_frame, fall_kpts, KEYPOINT_COLORS)

        # G. 화면 텍스트 출력
        rail_0 = rail_kpts.get('Rail_0')
        rail_1 = rail_kpts.get('Rail_1')

        cv2.putText(display_frame, f"In Bed: {in_bed_status} ({in_bed_method})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Pose: {pose_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"bed {bed.get('source', 'none')}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Rail_0: 1", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Rail_1: 0", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Fall: {fall_state} ({fall_score:.2f})", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255) if fall_alert else (255, 255, 0), 2)

        logging.info(
            f"In Bed: {in_bed_status} ({in_bed_method}) | Pose: {pose_label} | "
            f"bed {bed.get('source')} | "
            f"Rail_0: {rail_0} | Rail_1: {rail_1} | "
            f"Fall: {fall_state} ({fall_score:.2f})"
        )

        if use_gui:
            cv2.imshow("Smart Bed Monitoring", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if use_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = MyConfig()
    main(config)
