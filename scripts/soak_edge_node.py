#!/usr/bin/env python3
"""Poll an authenticated edge registry and summarize one node's heartbeat."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(url: str, token: str) -> dict:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    with urlopen(request, timeout=3.0) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8020/edge/nodes")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    started = time.monotonic()
    samples, errors = [], []
    while time.monotonic() - started < args.duration:
        tick = time.monotonic()
        try:
            payload = fetch(args.url, token)
            node = next(item for item in payload["nodes"] if item["node_id"] == args.node_id)
            heartbeat = node["heartbeat"]
            sent = datetime.fromisoformat(heartbeat["sent_at"].replace("Z", "+00:00"))
            samples.append({
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "sequence": heartbeat["sequence"],
                "heartbeat_age_sec": (datetime.now(timezone.utc) - sent).total_seconds(),
                "capture_connected": heartbeat["capture_connected"],
                "watcher_fps": heartbeat["watcher_fps"],
                "runtime_mode": heartbeat["runtime_mode"],
                "spool_depth": heartbeat["spool_depth"],
            })
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        remaining = args.interval - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)
    sequences = [row["sequence"] for row in samples]
    fps = [float(row["watcher_fps"]) for row in samples]
    report = {
        "schema_version": 1,
        "node_id": args.node_id,
        "duration_sec": args.duration,
        "sample_count": len(samples),
        "error_count": len(errors),
        "errors": errors,
        "summary": {
            "sequence_min": min(sequences) if sequences else None,
            "sequence_max": max(sequences) if sequences else None,
            "unique_sequences": len(set(sequences)),
            "capture_connected_all": all(row["capture_connected"] for row in samples),
            "watcher_fps_min": min(fps) if fps else None,
            "watcher_fps_mean": statistics.fmean(fps) if fps else None,
            "spool_depth_max": max((row["spool_depth"] for row in samples), default=None),
            "heartbeat_age_max_sec": max((row["heartbeat_age_sec"] for row in samples), default=None),
            "runtime_mode_counts": dict(Counter(row["runtime_mode"] for row in samples)),
        },
        "samples": samples,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
