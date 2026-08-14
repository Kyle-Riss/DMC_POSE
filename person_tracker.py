"""Lightweight multi-person pose tracking and primary patient selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class PersonDetection:
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    bbox: tuple[float, float, float, float]
    confidence: float
    bed_overlap: float


@dataclass
class PersonTrack:
    track_id: int
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    bbox: tuple[float, float, float, float]
    confidence_ema: float
    bed_overlap_ema: float
    first_seen_ts: float
    last_seen_ts: float
    hits: int = 1
    observed_this_frame: bool = True


@dataclass
class TrackingResult:
    tracks: list[PersonTrack]
    primary: PersonTrack | None
    primary_track_id: int | None
    primary_switched: bool
    expired_track_ids: list[int]


def keypoints_bbox(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    *,
    min_conf: float = 0.2,
) -> tuple[float, float, float, float] | None:
    xy = np.asarray(keypoints_xy, dtype=np.float32)
    conf = np.asarray(keypoints_conf, dtype=np.float32)
    valid = conf >= min_conf
    if np.count_nonzero(valid) < 3:
        return None
    points = xy[valid]
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    if x2 <= x1 or y2 <= y1:
        return None
    return float(x1), float(y1), float(x2), float(y2)


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


class MultiPersonTracker:
    """Greedy pose tracker with primary-patient hysteresis.

    This is intentionally CPU-only. It associates YOLO pose detections using
    bbox IoU and frame-normalized center distance, then favors a continuous
    track that has sustained bed overlap.
    """

    def __init__(
        self,
        *,
        track_ttl_sec: float = 2.0,
        max_center_distance_ratio: float = 0.25,
        primary_switch_margin: float = 0.25,
        ema_alpha: float = 0.35,
    ):
        self.track_ttl_sec = float(track_ttl_sec)
        self.max_center_distance_ratio = float(max_center_distance_ratio)
        self.primary_switch_margin = float(primary_switch_margin)
        self.ema_alpha = float(ema_alpha)
        self.tracks: dict[int, PersonTrack] = {}
        self.primary_track_id: int | None = None
        self.track_switch_total = 0
        self.track_created_total = 0
        self.track_expired_total = 0
        self._next_track_id = 1

    def _association_score(self, track, detection, frame_diag: float) -> float | None:
        iou = bbox_iou(track.bbox, detection.bbox)
        tx, ty = _center(track.bbox)
        dx, dy = _center(detection.bbox)
        distance_ratio = math.hypot(tx - dx, ty - dy) / max(1.0, frame_diag)
        if iou < 0.02 and distance_ratio > self.max_center_distance_ratio:
            return None
        return 2.0 * iou - distance_ratio

    def _primary_score(self, track: PersonTrack) -> float:
        continuity = 0.30 if track.track_id == self.primary_track_id else 0.0
        maturity = min(1.0, track.hits / 5.0)
        return (
            0.55 * track.bed_overlap_ema
            + 0.20 * track.confidence_ema
            + 0.15 * maturity
            + continuity
        )

    def update(
        self,
        detections: list[PersonDetection],
        timestamp: float,
        *,
        frame_width: int,
        frame_height: int,
    ) -> TrackingResult:
        timestamp = float(timestamp)
        for track in self.tracks.values():
            track.observed_this_frame = False

        expired = [
            track_id for track_id, track in self.tracks.items()
            if timestamp - track.last_seen_ts > self.track_ttl_sec
        ]
        for track_id in expired:
            self.tracks.pop(track_id, None)
        self.track_expired_total += len(expired)

        frame_diag = math.hypot(frame_width, frame_height)
        pairs = []
        for track_id, track in self.tracks.items():
            for detection_idx, detection in enumerate(detections):
                score = self._association_score(track, detection, frame_diag)
                if score is not None:
                    pairs.append((score, track_id, detection_idx))
        pairs.sort(reverse=True)
        matched_tracks = set()
        matched_detections = set()
        for _, track_id, detection_idx in pairs:
            if track_id in matched_tracks or detection_idx in matched_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[detection_idx]
            alpha = self.ema_alpha
            track.keypoints_xy = detection.keypoints_xy
            track.keypoints_conf = detection.keypoints_conf
            track.bbox = detection.bbox
            track.confidence_ema = (1 - alpha) * track.confidence_ema + alpha * detection.confidence
            track.bed_overlap_ema = (1 - alpha) * track.bed_overlap_ema + alpha * detection.bed_overlap
            track.last_seen_ts = timestamp
            track.hits += 1
            track.observed_this_frame = True
            matched_tracks.add(track_id)
            matched_detections.add(detection_idx)

        for detection_idx, detection in enumerate(detections):
            if detection_idx in matched_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self.track_created_total += 1
            self.tracks[track_id] = PersonTrack(
                track_id=track_id,
                keypoints_xy=detection.keypoints_xy,
                keypoints_conf=detection.keypoints_conf,
                bbox=detection.bbox,
                confidence_ema=detection.confidence,
                bed_overlap_ema=detection.bed_overlap,
                first_seen_ts=timestamp,
                last_seen_ts=timestamp,
            )

        previous_primary = self.primary_track_id
        active = list(self.tracks.values())
        if not active:
            self.primary_track_id = None
        elif self.primary_track_id not in self.tracks:
            self.primary_track_id = max(active, key=self._primary_score).track_id
        else:
            current = self.tracks[self.primary_track_id]
            challenger = max(active, key=self._primary_score)
            if (
                challenger.track_id != current.track_id
                and self._primary_score(challenger)
                > self._primary_score(current) + self.primary_switch_margin
            ):
                self.primary_track_id = challenger.track_id

        switched = (
            previous_primary is not None
            and self.primary_track_id is not None
            and previous_primary != self.primary_track_id
        )
        if switched:
            self.track_switch_total += 1
        primary = self.tracks.get(self.primary_track_id)
        if primary is not None and not primary.observed_this_frame:
            primary = None
        return TrackingResult(
            tracks=sorted(self.tracks.values(), key=lambda track: track.track_id),
            primary=primary,
            primary_track_id=self.primary_track_id,
            primary_switched=switched,
            expired_track_ids=expired,
        )

    def status(self) -> dict[str, Any]:
        primary = self.tracks.get(self.primary_track_id)
        return {
            "track_count": len(self.tracks),
            "primary_track_id": self.primary_track_id,
            "track_switch_total": self.track_switch_total,
            "track_created_total": self.track_created_total,
            "track_expired_total": self.track_expired_total,
            "primary_observed": bool(primary and primary.observed_this_frame),
            "primary_bed_overlap_ema": (
                float(primary.bed_overlap_ema) if primary is not None else 0.0
            ),
            "primary_confidence_ema": (
                float(primary.confidence_ema) if primary is not None else 0.0
            ),
        }
