"""Long-running, feature-only operational soak monitor."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any
from urllib.request import urlopen

from runtime_health import evaluate_fleet


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _numbers(samples: list[dict], camera_id: str, field: str) -> list[float]:
    values = []
    for sample in samples:
        value = sample.get("status", {}).get(camera_id, {}).get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _counter_delta(values: list[float]) -> int:
    if len(values) < 2:
        return 0
    # Counters reset when a process restarts. Sum each monotonic segment.
    total = 0.0
    previous = values[0]
    for current in values[1:]:
        total += current - previous if current >= previous else current
        previous = current
    return int(total)


def build_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    camera_ids = sorted({
        camera_id
        for sample in samples
        for camera_id in sample.get("status", {})
    })
    cameras = {}
    for camera_id in camera_ids:
        health_counts = Counter()
        for sample in samples:
            state = sample.get("status", {}).get(camera_id)
            if state is not None:
                health_counts[evaluate_fleet({camera_id: state})["status"]] += 1
        capture_fps = _numbers(samples, camera_id, "capture_fps")
        capture_age = _numbers(samples, camera_id, "capture_frame_age_ms")
        analysis_age = _numbers(samples, camera_id, "analysis_frame_age_ms")
        watcher_fps = _numbers(samples, camera_id, "watcher_fps")
        pending = _numbers(samples, camera_id, "scheduler_pending")
        cameras[camera_id] = {
            "samples": sum(health_counts.values()),
            "health_counts": dict(health_counts),
            "capture_fps_min": min(capture_fps) if capture_fps else None,
            "capture_fps_mean": statistics.fmean(capture_fps) if capture_fps else None,
            "capture_frame_age_ms_p95": _percentile(capture_age, 0.95),
            "analysis_frame_age_ms_p95": _percentile(analysis_age, 0.95),
            "watcher_fps_min": min(watcher_fps) if watcher_fps else None,
            "scheduler_pending_max": int(max(pending)) if pending else 0,
            "decode_errors_delta": _counter_delta(
                _numbers(samples, camera_id, "capture_decode_error_total")
            ),
            "reconnects_delta": _counter_delta(
                _numbers(samples, camera_id, "capture_reconnect_total")
            ),
            "scheduler_errors_delta": _counter_delta(
                _numbers(samples, camera_id, "scheduler_error_total")
            ),
            "scheduler_timeouts_delta": _counter_delta(
                _numbers(samples, camera_id, "scheduler_timeout_total")
            ),
            "stale_drops_delta": _counter_delta(
                _numbers(samples, camera_id, "scheduler_stale_drop_total")
            ),
        }
    ready_samples = sum(
        1 for sample in samples if sample.get("ready", {}).get("ready") is True
    )
    return {
        "schema_version": "dmc_pose_phase10_soak_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "started_at": samples[0]["sampled_at"] if samples else None,
        "ended_at": samples[-1]["sampled_at"] if samples else None,
        "http_error_count": sum(1 for sample in samples if sample.get("errors")),
        "process_ready_samples": ready_samples,
        "process_ready_ratio": ready_samples / len(samples) if samples else None,
        "cameras": cameras,
    }


def _get_json(base_url: str, path: str, timeout: float) -> dict:
    with urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"soak_{stamp}.jsonl"
    summary_path = output_dir / f"soak_{stamp}_summary.json"
    samples: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(0.0, args.duration_sec)

    while True:
        sample = {
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "sample_mono": time.monotonic(),
            "status": {},
            "ready": {},
            "recorder": {},
            "errors": {},
        }
        for key, path in (
            ("status", "/status"),
            ("ready", "/health/ready"),
            ("recorder", "/recorder/status"),
        ):
            try:
                sample[key] = _get_json(args.base_url, path, args.timeout_sec)
            except Exception as exc:
                sample["errors"][key] = f"{type(exc).__name__}: {exc}"
        samples.append(sample)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        summary = build_summary(samples)
        tmp = summary_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(summary_path)
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.1, min(args.interval_sec, deadline - time.monotonic())))
    return jsonl_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration-sec", type=float, default=3600.0)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        default="runtime_data/phase10_soak",
    )
    return parser.parse_args()


if __name__ == "__main__":
    paths = run(parse_args())
    print("\n".join(str(path) for path in paths))
