"""Automatic, multi-frame bed ROI stabilization and cache management."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from threading import Lock

import cv2
import numpy as np


def mask_to_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None or mask.size == 0:
        return None
    binary = mask > 0
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any() or not cols.any():
        return None
    return (
        int(np.argmax(cols)),
        int(np.argmax(rows)),
        int(len(cols) - np.argmax(cols[::-1]) - 1),
        int(len(rows) - np.argmax(rows[::-1]) - 1),
    )


def bbox_iou(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
) -> float:
    if left is None or right is None:
        return 0.0
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    left_area = max(0, left[2] - left[0] + 1) * max(0, left[3] - left[1] + 1)
    right_area = max(0, right[2] - right[0] + 1) * max(0, right[3] - right[1] + 1)
    union = left_area + right_area - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


class AutoBedROIManager:
    """Build a camera-specific bed ROI without a manually drawn region."""

    CACHE_SCHEMA_VERSION = 1

    def __init__(
        self,
        camera_id: str,
        cache_dir: Path,
        *,
        sample_every_n: int = 3,
        candidate_window: int = 5,
        min_detections: int = 3,
        consensus_iou: float = 0.75,
        refresh_sec: float = 300.0,
        scene_check_sec: float = 1.0,
        scene_pixel_threshold: int = 35,
        scene_change_ratio: float = 0.75,
        scene_change_persistence: int = 3,
    ):
        if min_detections < 2:
            raise ValueError("min_detections must be at least 2")
        if candidate_window < min_detections:
            raise ValueError("candidate_window must be >= min_detections")
        self.camera_id = str(camera_id)
        self.cache_dir = Path(cache_dir)
        self.sample_every_n = max(1, int(sample_every_n))
        self.candidate_window = int(candidate_window)
        self.min_detections = int(min_detections)
        self.consensus_iou = float(consensus_iou)
        self.refresh_sec = max(1.0, float(refresh_sec))
        self.scene_check_sec = max(0.1, float(scene_check_sec))
        self.scene_pixel_threshold = int(scene_pixel_threshold)
        self.scene_change_ratio = float(scene_change_ratio)
        self.scene_change_persistence = max(1, int(scene_change_persistence))

        self._lock = Lock()
        self._candidates = deque(maxlen=self.candidate_window)
        self._bed: dict | None = None
        self._stable = False
        self._version = 0
        self._agreement_iou = 0.0
        self._last_seg_attempt_mono: float | None = None
        self._last_detection_wall: float | None = None
        self._seg_run_count = 0
        self._invalid_reason = "no_auto_roi"
        self._scene_reference: np.ndarray | None = None
        self._last_scene_check_mono: float | None = None
        self._scene_change_count = 0
        self._frame_shape: tuple[int, int] | None = None
        self._load_cache()

    @property
    def json_path(self) -> Path:
        return self.cache_dir / f"{self.camera_id}.json"

    @property
    def mask_path(self) -> Path:
        return self.cache_dir / f"{self.camera_id}.mask.png"

    @property
    def scene_path(self) -> Path:
        return self.cache_dir / f"{self.camera_id}.scene.png"

    @staticmethod
    def _scene_thumbnail(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (80, 45), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _normalize_detection(detection: dict, frame_shape: tuple[int, int]) -> dict | None:
        h, w = frame_shape
        mask = detection.get("mask")
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)
            if not mask.any():
                mask = None
        bbox = detection.get("bbox")
        if bbox is not None:
            bbox = tuple(int(value) for value in bbox)
        if bbox is None:
            bbox = mask_to_bbox(mask)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        bbox = (
            int(np.clip(x1, 0, w - 1)),
            int(np.clip(y1, 0, h - 1)),
            int(np.clip(x2, 0, w - 1)),
            int(np.clip(y2, 0, h - 1)),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return {
            "mask": mask.copy() if mask is not None else None,
            "raw_mask": mask.copy() if mask is not None else None,
            "bbox": bbox,
            "confidence": float(detection.get("confidence", 0.0)),
            "source": "auto_candidate",
        }

    def should_run_segmentation(self, frame_seq: int, mono_ts: float) -> bool:
        with self._lock:
            if not self._stable:
                return (int(frame_seq) - 1) % self.sample_every_n == 0
            if self._last_seg_attempt_mono is None:
                return True
            return float(mono_ts) - self._last_seg_attempt_mono >= self.refresh_sec

    def mark_segmentation_attempt(self, mono_ts: float) -> None:
        with self._lock:
            self._last_seg_attempt_mono = float(mono_ts)
            self._seg_run_count += 1

    def observe_frame(self, frame: np.ndarray, mono_ts: float) -> bool:
        """Return True when a stable/cache ROI was invalidated by scene change."""
        h, w = frame.shape[:2]
        with self._lock:
            if self._frame_shape is None:
                self._frame_shape = (h, w)
            elif self._frame_shape != (h, w):
                self._invalidate_locked("frame_shape_changed")
                self._frame_shape = (h, w)
                return True

            if not self._stable or self._scene_reference is None:
                return False
            if (
                self._last_scene_check_mono is not None
                and float(mono_ts) - self._last_scene_check_mono < self.scene_check_sec
            ):
                return False
            self._last_scene_check_mono = float(mono_ts)
            current = self._scene_thumbnail(frame)
            changed = cv2.absdiff(self._scene_reference, current) >= self.scene_pixel_threshold
            ratio = float(np.count_nonzero(changed)) / float(changed.size)
            if ratio >= self.scene_change_ratio:
                self._scene_change_count += 1
            else:
                self._scene_change_count = 0
            if self._scene_change_count >= self.scene_change_persistence:
                self._invalidate_locked("scene_changed")
                return True
            return False

    def observe_detection(
        self,
        detection: dict,
        frame: np.ndarray,
        *,
        mono_ts: float,
        wall_ts: float,
    ) -> bool:
        """Consume one segmentation result and return current stable status."""
        h, w = frame.shape[:2]
        normalized = self._normalize_detection(detection, (h, w))
        with self._lock:
            self._last_seg_attempt_mono = float(mono_ts)
            self._frame_shape = (h, w)
            if normalized is None:
                return self._stable

            if self._stable and self._bed is not None:
                agreement = bbox_iou(self._bed.get("bbox"), normalized.get("bbox"))
                if agreement >= self.consensus_iou:
                    self._bed = {
                        **normalized,
                        "source": "auto_refresh",
                    }
                    self._agreement_iou = agreement
                    self._last_detection_wall = float(wall_ts)
                    self._scene_reference = self._scene_thumbnail(frame)
                    self._invalid_reason = ""
                    self._persist_locked(valid=True)
                    return True
                self._invalidate_locked("refresh_disagreed")

            self._candidates.append(normalized)
            if len(self._candidates) < self.min_detections:
                self._bed = {**normalized, "source": "auto_provisional"}
                self._invalid_reason = "collecting_consensus"
                return False

            recent = list(self._candidates)
            pairwise = [
                bbox_iou(recent[left]["bbox"], recent[right]["bbox"])
                for left in range(len(recent))
                for right in range(left + 1, len(recent))
            ]
            agreement = float(np.median(pairwise)) if pairwise else 0.0
            self._agreement_iou = agreement
            if agreement < self.consensus_iou:
                self._bed = {**normalized, "source": "auto_provisional"}
                self._invalid_reason = "consensus_not_reached"
                return False

            masks = [candidate["mask"] for candidate in recent if candidate["mask"] is not None]
            consensus_mask = None
            if len(masks) >= self.min_detections:
                votes = np.stack(masks, axis=0).sum(axis=0)
                consensus_mask = (
                    votes >= math.ceil(len(masks) / 2.0)
                ).astype(np.uint8)
            if consensus_mask is not None and consensus_mask.any():
                consensus_bbox = mask_to_bbox(consensus_mask)
            else:
                coords = np.asarray([candidate["bbox"] for candidate in recent], dtype=np.float32)
                consensus_bbox = tuple(int(round(value)) for value in np.median(coords, axis=0))

            confidence = float(np.median([
                candidate["confidence"] for candidate in recent
            ]))
            self._version += 1
            self._bed = {
                "mask": consensus_mask,
                "raw_mask": consensus_mask.copy() if consensus_mask is not None else None,
                "bbox": consensus_bbox,
                "confidence": confidence,
                "source": "auto_consensus",
            }
            self._stable = True
            self._last_detection_wall = float(wall_ts)
            self._scene_reference = self._scene_thumbnail(frame)
            self._scene_change_count = 0
            self._invalid_reason = ""
            self._candidates.clear()
            self._persist_locked(valid=True)
            return True

    def current_bbox(self) -> tuple[int, int, int, int] | None:
        """Return only the stable bbox without copying full-size masks."""
        with self._lock:
            if not self._stable or self._bed is None:
                return None
            bbox = self._bed.get("bbox")
            return tuple(int(value) for value in bbox) if bbox is not None else None

    def current(self) -> dict:
        with self._lock:
            # Provisional detections are visible through status(), but must not
            # affect safety decisions until multi-frame consensus is complete.
            if not self._stable or self._bed is None:
                return {
                    "mask": None,
                    "raw_mask": None,
                    "bbox": None,
                    "confidence": 0.0,
                    "source": "auto_not_ready",
                }
            result = dict(self._bed)
            if result.get("mask") is not None:
                result["mask"] = result["mask"].copy()
            if result.get("raw_mask") is not None:
                result["raw_mask"] = result["raw_mask"].copy()
            return result

    def status(self) -> dict:
        with self._lock:
            source = self._bed.get("source", "auto_not_ready") if self._bed else "auto_not_ready"
            roi_state = "READY" if self._stable else ("DEGRADED" if self._bed else "NOT_READY")
            return {
                "ready": self._stable,
                "roi_state": roi_state,
                "restored_from_cache": source == "auto_cache",
                "version": self._version,
                "source": source,
                "confidence": float(self._bed.get("confidence", 0.0)) if self._bed else 0.0,
                "agreement_iou": self._agreement_iou,
                "candidate_count": len(self._candidates),
                "seg_run_count": self._seg_run_count,
                "last_detection_wall": self._last_detection_wall,
                "invalid_reason": self._invalid_reason,
            }

    def _invalidate_locked(self, reason: str) -> None:
        self._stable = False
        self._bed = None
        self._candidates.clear()
        self._agreement_iou = 0.0
        self._scene_reference = None
        self._scene_change_count = 0
        self._invalid_reason = str(reason)
        self._persist_locked(valid=False)

    def _persist_locked(self, *, valid: bool) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "camera_id": self.camera_id,
            "valid": bool(valid),
            "version": self._version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "frame_height": self._frame_shape[0] if self._frame_shape else None,
            "frame_width": self._frame_shape[1] if self._frame_shape else None,
            "bbox": list(self._bed["bbox"]) if valid and self._bed and self._bed.get("bbox") else None,
            "confidence": float(self._bed.get("confidence", 0.0)) if valid and self._bed else 0.0,
            "agreement_iou": self._agreement_iou,
            "invalid_reason": "" if valid else self._invalid_reason,
        }
        if valid and self._bed is not None and self._bed.get("mask") is not None:
            mask_tmp = self.mask_path.with_name(f"{self.mask_path.stem}.tmp.png")
            if not cv2.imwrite(str(mask_tmp), self._bed["mask"] * 255):
                raise OSError(f"failed to write automatic bed mask: {mask_tmp}")
            os.replace(mask_tmp, self.mask_path)
        if valid and self._scene_reference is not None:
            scene_tmp = self.scene_path.with_name(f"{self.scene_path.stem}.tmp.png")
            if not cv2.imwrite(str(scene_tmp), self._scene_reference):
                raise OSError(f"failed to write scene reference: {scene_tmp}")
            os.replace(scene_tmp, self.scene_path)
        json_tmp = self.json_path.with_suffix(".tmp.json")
        json_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(json_tmp, self.json_path)

    def _load_cache(self) -> None:
        if not self.json_path.is_file():
            return
        try:
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") != self.CACHE_SCHEMA_VERSION
                or payload.get("camera_id") != self.camera_id
                or not payload.get("valid")
            ):
                return
            h = int(payload["frame_height"])
            w = int(payload["frame_width"])
            bbox = tuple(int(value) for value in payload["bbox"])
            mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != (h, w):
                    return
                mask = (mask > 0).astype(np.uint8)
            scene = cv2.imread(str(self.scene_path), cv2.IMREAD_GRAYSCALE)
            if scene is not None and scene.shape != (45, 80):
                scene = None
            self._bed = {
                "mask": mask,
                "raw_mask": mask.copy() if mask is not None else None,
                "bbox": bbox,
                "confidence": float(payload.get("confidence", 0.0)),
                "source": "auto_cache",
            }
            self._stable = True
            self._version = int(payload.get("version", 1))
            self._agreement_iou = float(payload.get("agreement_iou", 0.0))
            self._frame_shape = (h, w)
            self._scene_reference = scene
            self._last_detection_wall = None
            self._invalid_reason = ""
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._bed = None
            self._stable = False
            self._invalid_reason = "auto_cache_invalid"

