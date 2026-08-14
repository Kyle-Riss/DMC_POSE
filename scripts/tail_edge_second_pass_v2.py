#!/usr/bin/env python3
"""Event-only second-pass audit tailer with project import bootstrap."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from edge_contract_v1 import EdgeEventStart  # noqa: E402
from selective_second_pass import SelectiveSecondPass  # noqa: E402
from central_canary_second_pass import CentralCanarySecondPass  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "runtime_data/edge_control/edge_control.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runtime_data/edge_second_pass")
    parser.add_argument("--frame-root", type=Path, default=PROJECT / "runtime_data/edge_event_frames")
    parser.add_argument("--reference-status", default="http://127.0.0.1:8000/status")
    parser.add_argument("--from-start", action="store_true")
    args = parser.parse_args()
    verifier = CentralCanarySecondPass(args.frame_root, status_url=args.reference_status)
    dispatcher = SelectiveSecondPass(verifier, args.output_dir, queue_size=32, workers=1)
    dispatcher.start()
    args.input.parent.mkdir(parents=True, exist_ok=True)
    args.input.touch(exist_ok=True)
    try:
        with args.input.open(encoding="utf-8") as handle:
            if not args.from_start:
                handle.seek(0, 2)
            while True:
                line = handle.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                try:
                    item = json.loads(line)
                    if item.get("kind") == "event_start":
                        dispatcher.submit(EdgeEventStart.model_validate(item["payload"]))
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
    except KeyboardInterrupt:
        return 0
    finally:
        dispatcher.stop()


if __name__ == "__main__":
    raise SystemExit(main())
