import numpy as np

from edge_motion_watcher import EdgeMotionWatcher


def watcher():
    return EdgeMotionWatcher(
        "rtsp://127.0.0.1:8554/stream",
        width=64,
        height=36,
        pixel_threshold=10,
        motion_ratio_threshold=0.02,
        consecutive_hits=2,
        burst_hold_sec=1.0,
    )


def test_two_valid_hits_open_and_expire_burst():
    item = watcher()
    frames = []
    for x in (0, 10, 20):
        frame = np.zeros((36, 64), dtype=np.uint8)
        frame[8:28, x:x + 20] = 255
        frames.append(frame)
    item.process_gray(frames[0], mono_ts=1.0)
    assert item.process_gray(frames[1], mono_ts=1.1)["burst_active"] is False
    status = item.process_gray(frames[2], mono_ts=1.2)
    assert status["burst_active"] is True
    assert status["trigger_total"] == 1
    assert item.status(now=2.21)["burst_active"] is False


def test_full_frame_glitch_is_rejected():
    item = watcher()
    dark = np.zeros((36, 64), dtype=np.uint8)
    bright = np.full((36, 64), 255, dtype=np.uint8)
    item.process_gray(dark, mono_ts=1.0)
    status = item.process_gray(bright, mono_ts=1.1)
    assert status["motion_hit_streak"] == 0
    assert status["burst_active"] is False


def test_status_reports_observed_fps():
    item = watcher()
    frame = np.zeros((36, 64), dtype=np.uint8)
    for timestamp in (1.0, 1.2, 1.4):
        item.process_gray(frame, mono_ts=timestamp)
    assert abs(item.status(now=1.4)["watcher_fps"] - 5.0) < 1e-9



def test_latest_rgb_is_defensive_copy():
    item = EdgeMotionWatcher("rtsp://example", retain_rgb=True)
    frame = np.ones((180, 320, 3), dtype=np.uint8)
    with item._lock:
        item._latest_rgb = frame.copy()
        item._latest_rgb_ts = 1.0
    first = item.latest_rgb()
    first[:] = 0
    assert np.all(item.latest_rgb() == 1)
