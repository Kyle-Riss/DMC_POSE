import json
import os

import numpy as np

from edge_site_runtime import (
    EdgeROIProfile,
    EdgeSceneGuard,
    EdgeSiteRuntime,
    EncodedRingMonitor,
)


def rgb(value):
    return np.full((180, 320, 3), value, dtype=np.uint8)


def test_scene_guard_calibrates_persists_and_latches_change(tmp_path):
    path = tmp_path / "scene.bin"
    guard = EdgeSceneGuard(path, change_ratio=0.5, persistence=2)
    assert guard.status()["state"] == "UNCALIBRATED"
    guard.calibrate(rgb(0))
    assert path.is_file()
    assert guard.observe(rgb(255), wall_ts=1.0)["state"] == "STABLE"
    assert guard.observe(rgb(255), wall_ts=2.0)["state"] == "CHANGED"
    restored = EdgeSceneGuard(path)
    assert restored.status()["state"] == "STABLE"


def test_roi_is_degraded_when_scene_changes(tmp_path):
    path = tmp_path / "roi.json"
    path.write_text(json.dumps({
        "version": 3,
        "source": "aruco_setup",
        "polygon_norm": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]],
    }))
    roi = EdgeROIProfile(path)
    assert roi.status("STABLE")["state"] == "READY"
    assert roi.status("CHANGED")["state"] == "DEGRADED"
    assert roi.status("CHANGED")["valid"] is False


def test_ring_monitor_reports_encoded_coverage_without_deleting(tmp_path):
    first = tmp_path / "0001.ts"
    second = tmp_path / "0002.ts"
    first.write_bytes(b"a" * 10)
    second.write_bytes(b"b" * 20)
    os.utime(first, (90.0, 90.0))
    os.utime(second, (98.0, 98.0))
    monitor = EncodedRingMonitor(
        tmp_path, segment_duration_sec=2.0, maximum_segment_age_sec=5.0,
        wall_clock=lambda: 100.0,
    )
    status = monitor.status()
    assert status["ready"] is True
    assert status["segments"] == 2
    assert status["bytes"] == 30
    assert status["coverage_sec"] == 10.0
    assert first.exists() and second.exists()


def test_site_runtime_reuses_watcher_snapshot_and_never_opens_rtsp(tmp_path):
    class Watcher:
        def status(self):
            return {"motion_ratio": 0.12, "burst_active": True}

        def latest_rgb_snapshot(self):
            return rgb(0), 10.0

    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps({
        "version": 1,
        "source": "fixed_profile",
        "polygon_norm": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]],
    }))
    runtime = EdgeSiteRuntime({
        "scene_guard": {"reference_path": str(tmp_path / "scene.bin")},
        "roi_profile_path": str(roi_path),
    }, Watcher())
    runtime.calibrate_from_latest()
    status = runtime.refresh()
    assert status["motion_active"] is True
    assert status["scene_state"] == "STABLE"
    assert status["roi_state"] == "READY"
