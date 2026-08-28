#!/usr/bin/env python3
"""Build a human-review queue from legacy AI_runner automatic fall events."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

EVENT_RE = re.compile(r"^(bed_[^_]+)_fall_(\d{8})_(\d{6})$")
FIELDS = (
    "event_id", "cluster_id", "event_time", "bed_id", "frame_count",
    "duration_sec", "last_confidence", "mean_positive_confidence",
    "positive_samples", "bbox_center_drop", "review_label", "review_status",
    "reviewer", "review_notes", "event_dir", "contact_sheet",
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def event_time(path: Path) -> tuple[str, datetime] | None:
    match = EVENT_RE.match(path.name)
    if not match:
        return None
    return match.group(1), datetime.strptime("".join(match.groups()[1:]), "%Y%m%d%H%M%S")


def event_record(path: Path) -> dict | None:
    parsed = event_time(path)
    if parsed is None:
        return None
    bed_id, timestamp = parsed
    meta = read_json(path / "event_meta.json")
    result = read_json(path / "fall_result.json").get("result", {})
    samples = result.get("samples") or []
    frames = sorted((path / "frames").glob("*.jpg"))
    confidences = [float(row["confidence"]) for row in samples if row.get("positive") and row.get("confidence") is not None]
    boxes = [row.get("boundingBox") for row in samples if row.get("boundingBox") and len(row["boundingBox"]) == 4]
    centers = [(float(box[1]) + float(box[3])) / 2.0 for box in boxes]
    ring = meta.get("ring_buffer") or {}
    first_ts, last_ts = ring.get("first_timestamp"), ring.get("last_timestamp")
    duration = float(last_ts) - float(first_ts) if first_ts is not None and last_ts is not None else 0.0
    last_result = result.get("lastResult") or {}
    return {
        "event_id": path.name,
        "event_time": timestamp,
        "bed_id": bed_id,
        "frame_count": len(frames),
        "duration_sec": duration,
        "last_confidence": float(last_result.get("confidence") or 0.0),
        "mean_positive_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "positive_samples": int(result.get("positiveSamples") or len(confidences)),
        "bbox_center_drop": centers[-1] - centers[0] if len(centers) >= 2 else 0.0,
        "event_dir": str(path.resolve()),
        "frames": frames,
    }


def assign_clusters(events: list[dict], gap_sec: float) -> None:
    events.sort(key=lambda row: row["event_time"])
    cluster_index = 0
    previous = None
    for row in events:
        if previous is None or (row["event_time"] - previous).total_seconds() > gap_sec:
            cluster_index += 1
        row["cluster_id"] = f"cluster_{cluster_index:04d}"
        previous = row["event_time"]


def contact_sheet(frames: list[Path], out: Path, samples: int, tile_width: int) -> None:
    if not frames:
        return
    indices = np.linspace(0, len(frames) - 1, min(samples, len(frames))).round().astype(int)
    tiles = []
    for index in indices:
        image = cv2.imread(str(frames[int(index)]))
        if image is None:
            continue
        scale = tile_width / image.shape[1]
        resized = cv2.resize(image, (tile_width, max(1, int(round(image.shape[0] * scale)))))
        cv2.rectangle(resized, (0, 0), (tile_width, 23), (0, 0, 0), -1)
        cv2.putText(resized, f"{int(index):03d}/{len(frames)-1:03d}", (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(resized)
    if not tiles:
        return
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    height, width = tiles[0].shape[:2]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        canvas[y * height : (y + 1) * height, x * width : (x + 1) * width] = tile
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 86]):
        raise OSError(f"cannot write {out}")


def write_html(rows: list[dict], out: Path) -> None:
    cards = []
    for row in rows:
        sheet = Path(row["contact_sheet"]).name
        cards.append(
            f'<article><h3>{row["event_id"]}</h3><img loading="lazy" src="sheets/{sheet}">'
            f'<p>{row["cluster_id"]} · conf {float(row["last_confidence"]):.3f} · '
            f'drop {float(row["bbox_center_drop"]):.1f}px · {row["frame_count"]} frames</p></article>'
        )
    out.write_text("""<!doctype html><meta charset="utf-8"><title>DMC fall review queue</title>
<style>body{font:14px sans-serif;background:#111;color:#eee}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px}article{background:#222;padding:10px;border-radius:8px}img{width:100%;height:auto}h3,p{margin:5px}</style>
<h1>AI_runner fall candidates — human review required</h1><p>Automatic confidence is not ground truth. Record labels in review_queue.csv.</p><main>""" + "\n".join(cards) + "</main>\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/dmc/AI/AI_runner/data/events/fall"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runtime_data/ai_runner_fall_review_v1")
    parser.add_argument("--cluster-gap-sec", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--tile-width", type=int, default=240)
    args = parser.parse_args()
    events = [row for path in args.root.iterdir() if path.is_dir() if (row := event_record(path))]
    assign_clusters(events, args.cluster_gap_sec)
    events.sort(key=lambda row: (-row["last_confidence"], -row["bbox_center_drop"], row["event_time"]))
    out_dir = args.out_dir.resolve()
    sheets_dir = out_dir / "sheets"
    for row in events:
        sheet = sheets_dir / f"{row['event_id']}.jpg"
        contact_sheet(row.pop("frames"), sheet, args.samples, args.tile_width)
        row["contact_sheet"] = str(sheet)
        row["event_time"] = row["event_time"].isoformat()
        row.update(review_label="", review_status="unreviewed", reviewer="", review_notes="")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "review_queue.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(events)
    summary = {
        "schema_version": "dmc_ai_runner_fall_review_queue_v1",
        "source_root": str(args.root.resolve()),
        "event_count": len(events),
        "cluster_count": len({row["cluster_id"] for row in events}),
        "label_contract": ["fall", "non_fall", "uncertain"],
        "warning": "Automatic candidates are not ground truth until human review is complete.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(events, out_dir / "index.html")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"review queue: {out_dir / 'review_queue.csv'}")
    print(f"browser index: {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
