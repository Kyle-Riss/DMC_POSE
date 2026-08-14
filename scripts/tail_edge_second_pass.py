#!/usr/bin/env python3
"""Tail edge audit JSONL and dispatch only eligible event starts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from edge_contract_v1 import EdgeEventStart
from selective_second_pass import SelectiveSecondPass


def metadata_verifier(event: EdgeEventStart) -> dict:
    """Safe pre-Pi mode; proves dispatch without opening a continuous stream."""
    return {
        "decision": "await_event_frames_or_rtsp_burst",
        "continuous_stream_opened": False,
        "event_frame_upload_requested": event.event_type in {"FALL", "BED_EXIT_FALL"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runtime_data/edge_control/edge_control.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime_data/edge_second_pass"))
    parser.add_argument("--from-start", action="store_true")
    args = parser.parse_args()
    dispatcher = SelectiveSecondPass(metadata_verifier, args.output_dir, queue_size=32, workers=1)
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
                except Exception:
                    continue
    except KeyboardInterrupt:
        return 0
    finally:
        dispatcher.stop()


if __name__ == "__main__":
    raise SystemExit(main())
