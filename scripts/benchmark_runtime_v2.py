#!/usr/bin/env python3
"""Measure status and MJPEG delivery without changing the running server."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from urllib.request import urlopen


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sample_mjpeg(base_url, camera_id, duration):
    started = time.monotonic()
    frames = 0
    carried = b""
    error = None
    try:
        with urlopen(f"{base_url}/video/{camera_id}", timeout=max(5.0, duration + 3.0)) as response:
            while time.monotonic() - started < duration:
                chunk = response.read(65536)
                if not chunk:
                    break
                data = carried + chunk
                frames += data.count(b"\xff\xd8")
                carried = data[-1:]
    except Exception as exc:
        error = str(exc)
    elapsed = max(1e-6, time.monotonic() - started)
    return {
        "frames": frames,
        "elapsed_sec": elapsed,
        "display_fps": frames / elapsed,
        "error": error,
    }


def status_samples(base_url, duration, interval):
    started = time.monotonic()
    samples = []
    errors = []
    while time.monotonic() - started < duration:
        tick = time.monotonic()
        try:
            with urlopen(f"{base_url}/status", timeout=3.0) as response:
                samples.append(json.load(response))
        except Exception as exc:
            errors.append(str(exc))
        remaining = interval - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)
    return samples, errors


def summarize_status(samples, camera_ids):
    result = {}
    for camera_id in camera_ids:
        rows = [sample[camera_id] for sample in samples if camera_id in sample]
        pipeline_fps = [row.get("pipeline_fps", 0.0) for row in rows]
        latency_ms = [row.get("latency_ms", 0.0) for row in rows]
        capture_fps = [
            row["capture_fps"] for row in rows
            if isinstance(row.get("capture_fps"), (int, float))
        ]
        capture_age = [
            row["capture_frame_age_ms"] for row in rows
            if isinstance(row.get("capture_frame_age_ms"), (int, float))
        ]
        result[camera_id] = {
            "sample_count": len(rows),
            "states": sorted({str(row.get("analysis_state", "unknown")) for row in rows}),
            "pipeline_fps_mean": statistics.fmean(pipeline_fps) if pipeline_fps else None,
            "pipeline_latency_ms_p95": percentile(latency_ms, 0.95),
            "capture_fps_mean": statistics.fmean(capture_fps) if capture_fps else None,
            "capture_frame_age_ms_p95": percentile(capture_age, 0.95),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--status-interval", type=float, default=0.5)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["bed_161", "bed_162", "bed_174", "bed_175", "bed_178", "bed_179"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with ThreadPoolExecutor(max_workers=len(args.cameras) + 1) as executor:
        status_future = executor.submit(
            status_samples, args.base_url.rstrip("/"), args.duration, args.status_interval
        )
        video_futures = {
            camera_id: executor.submit(
                sample_mjpeg, args.base_url.rstrip("/"), camera_id, args.duration
            )
            for camera_id in args.cameras
        }
        samples, status_errors = status_future.result()
        video = {
            camera_id: future.result()
            for camera_id, future in video_futures.items()
        }

    report = {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "duration_sec": args.duration,
        "status_errors": status_errors,
        "status": summarize_status(samples, args.cameras),
        "video": video,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

