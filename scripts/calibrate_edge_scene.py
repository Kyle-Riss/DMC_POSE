#!/usr/bin/env python3
"""Capture one local watcher frame and persist the fixed-camera scene baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_motion_watcher import EdgeMotionWatcher
from edge_site_runtime import EdgeSiteRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    watcher_config = dict(config.get("motion_watcher_config", {}))
    site_config = dict(config.get("site_runtime_config", {}))
    if not watcher_config.get("retain_rgb"):
        raise ValueError("motion_watcher_config.retain_rgb must be true")
    if not site_config.get("scene_guard"):
        raise ValueError("site_runtime_config.scene_guard is required")

    watcher = EdgeMotionWatcher(**watcher_config)
    runtime = EdgeSiteRuntime(site_config, watcher)
    deadline = time.monotonic() + max(1.0, args.timeout_sec)
    watcher.start()
    try:
        while watcher.latest_rgb_snapshot() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if watcher.latest_rgb_snapshot() is None:
            raise TimeoutError("no local RTSP frame arrived before calibration timeout")
        result = runtime.calibrate_from_latest()
    finally:
        watcher.stop()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
