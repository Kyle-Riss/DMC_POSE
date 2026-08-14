#!/usr/bin/env python3
"""Quick CLI to test rail detection on a still image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from rail_detect import detect_both_rails, load_rail_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"cannot read {args.image}")

    cfg = load_rail_config(args.config)
    result = detect_both_rails(frame, cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
