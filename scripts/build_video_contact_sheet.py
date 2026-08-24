#!/usr/bin/env python3
"""Build an exact-frame timestamped contact sheet for manual video review."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_contact_sheet(
    video: Path,
    out: Path,
    *,
    columns: int = 4,
    rows: int = 6,
    tile_width: int = 320,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0 or fps <= 0.0:
        capture.release()
        raise ValueError(f"invalid video metadata: {video}")
    if columns <= 0 or rows <= 0 or tile_width <= 0:
        capture.release()
        raise ValueError("columns, rows, and tile_width must be positive")
    resolved_end = frame_count - 1 if end_frame is None else end_frame
    if start_frame < 0 or resolved_end >= frame_count or start_frame > resolved_end:
        capture.release()
        raise ValueError(
            f"invalid review range [{start_frame}, {resolved_end}] for {frame_count} frames"
        )
    sample_count = min(columns * rows, resolved_end - start_frame + 1)
    indices = np.linspace(start_frame, resolved_end, sample_count).round().astype(int)
    wanted = {int(index) for index in indices}
    decoded: dict[int, np.ndarray] = {}
    last_decoded = -1
    for current in range(resolved_end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        last_decoded = current
        if current in wanted:
            decoded[current] = frame
    capture.release()
    missing = sorted(wanted - decoded.keys())
    if missing:
        raise ValueError(
            f"cannot decode requested frames {missing} from {video}; "
            f"last sequentially decoded frame is {last_decoded}"
        )
    tiles = []
    tile_height = None
    for index in indices:
        frame = decoded[int(index)]
        height = max(1, round(frame.shape[0] * tile_width / frame.shape[1]))
        tile = cv2.resize(frame, (tile_width, height), interpolation=cv2.INTER_AREA)
        tile_height = height
        label = f"frame {index}  {index / fps:.2f}s"
        cv2.rectangle(tile, (0, 0), (tile_width, 30), (0, 0, 0), -1)
        cv2.putText(tile, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for position, tile in enumerate(tiles):
        row, column = divmod(position, columns)
        canvas[row * tile_height:(row + 1) * tile_height, column * tile_width:(column + 1) * tile_width] = tile
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise ValueError(f"cannot write contact sheet: {out}")
    return {"video": str(video), "out": str(out), "fps": fps, "frame_count": frame_count, "sampled_frames": indices.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    result = build_contact_sheet(
        args.video.resolve(),
        args.out.resolve(),
        columns=args.columns,
        rows=args.rows,
        tile_width=args.tile_width,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
