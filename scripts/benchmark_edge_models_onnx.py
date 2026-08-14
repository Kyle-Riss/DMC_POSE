#!/usr/bin/env python3
"""Benchmark ONNX edge artifacts without opening a camera or RTSP stream."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def concrete_shape(shape: list[object]) -> tuple[int, ...]:
    return tuple(int(value) if isinstance(value, int) and value > 0 else 1 for value in shape)


def benchmark(path: Path, warmup: int, iterations: int) -> dict:
    load_started = time.perf_counter()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - load_started) * 1000
    feed = {
        item.name: np.random.default_rng(42).random(concrete_shape(item.shape), dtype=np.float32)
        for item in session.get_inputs()
    }
    for _ in range(warmup):
        session.run(None, feed)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        session.run(None, feed)
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "filename": path.name,
        "load_ms": load_ms,
        "iterations": iterations,
        "mean_ms": statistics.fmean(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "max_ms": max(samples),
        "throughput_fps": 1000.0 / statistics.fmean(samples),
        "inputs": [
            {"name": item.name, "shape": list(item.shape), "type": item.type}
            for item in session.get_inputs()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for artifact in manifest["artifacts"]:
        if artifact["format"] != "onnx":
            continue
        results[artifact["role"]] = benchmark(
            args.bundle / artifact["filename"], args.warmup, args.iterations
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_version": manifest["bundle_version"],
        "target": manifest["target"],
        "provider": "CPUExecutionProvider",
        "warmup": args.warmup,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
