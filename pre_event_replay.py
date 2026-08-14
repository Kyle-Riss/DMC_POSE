"""Replay pre-event pose detections without corrupting the live time axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from live_temporal import TemporalModelService, TemporalShadowRunner
from person_tracker import MultiPersonTracker, PersonDetection


@dataclass(frozen=True)
class ReplayPoseFrame:
    mono_ts: float
    frame_width: int
    frame_height: int
    detections: tuple[PersonDetection, ...]


def replay_temporal_context(
    frames: list[ReplayPoseFrame],
    service: TemporalModelService,
    classify_batch: Callable[[np.ndarray], np.ndarray],
    *,
    track_ttl_sec: float = 1.0,
    primary_switch_margin: float = 0.25,
) -> dict:
    """Build one temporary observed-only TCN context from chronological frames.

    The live tracker and live TCN runner are never touched.  This prevents
    historical timestamps from being inserted behind a current live sample.
    """
    ordered = sorted(frames, key=lambda item: item.mono_ts)
    tracker = MultiPersonTracker(
        track_ttl_sec=track_ttl_sec,
        primary_switch_margin=primary_switch_margin,
    )
    runner = TemporalShadowRunner(service)
    observations: list[tuple[ReplayPoseFrame, object | None, bool]] = []
    selected_xy: list[np.ndarray] = []

    for item in ordered:
        tracking = tracker.update(
            list(item.detections), item.mono_ts,
            frame_width=item.frame_width,
            frame_height=item.frame_height,
        )
        primary = tracking.primary
        observations.append((item, primary, tracking.primary_switched))
        if primary is not None:
            selected_xy.append(np.asarray(primary.keypoints_xy, dtype=np.float32))

    if selected_xy:
        classifier_input = np.stack(selected_xy).reshape(len(selected_xy), -1)
        probabilities = np.asarray(classify_batch(classifier_input), dtype=np.float32)
        if probabilities.shape != (len(selected_xy), 6):
            raise ValueError(
                "replay classifier must return "
                f"({len(selected_xy)}, 6), got {probabilities.shape}"
            )
    else:
        probabilities = np.empty((0, 6), dtype=np.float32)

    probability_index = 0
    observed_count = 0
    track_reset_total = 0
    last_ready_status: dict | None = None
    for item, primary, primary_switched in observations:
        if primary_switched:
            runner.reset()
            track_reset_total += 1
        if primary is None:
            runner.observe_gap(item.mono_ts)
            continue
        step_status = runner.push(
            item.mono_ts,
            primary.keypoints_xy,
            primary.keypoints_conf,
            probabilities[probability_index],
            timestamp_source="pre_event_ring_mono_ts",
        )
        if step_status.get("ready", False):
            # A later missing observation may reset only the tail segment.
            # Keep the most recent fully observed 30-row window found anywhere
            # in the bounded pre-event history.
            last_ready_status = dict(step_status)
        probability_index += 1
        observed_count += 1

    final_status = runner.status()
    status = dict(last_ready_status or final_status)
    status["latest_segment_samples"] = int(final_status.get("samples", 0))
    for counter in (
        "gap_reset_total", "duplicate_skip_total",
        "non_monotonic_skip_total", "prediction_count",
    ):
        status[counter] = final_status.get(counter, status.get(counter, 0))
    status.update({
        "requested_frames": len(ordered),
        "observed_pose_frames": observed_count,
        "track_reset_total": track_reset_total,
        "track_created_total": tracker.track_created_total,
        "source": "pre_event_replay",
    })
    return status
