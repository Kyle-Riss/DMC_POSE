#!/usr/bin/env python3
"""Draw bbox labels on a frame (extended rail / folded side)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

LABELS = ("extended rail", "folded side")
COLORS = {
    "extended rail": (0, 255, 0),
    "folded side": (0, 165, 255),
}


def draw_boxes(img, boxes: list[dict]) -> None:
    for b in boxes:
        label = b["label"]
        x0, y0, x1, y1 = [int(v) for v in b["bbox"]]
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, label, (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate rail bboxes on a still frame")
    ap.add_argument("image", type=Path, help="input image path")
    ap.add_argument("--out", type=Path, help="annotated image (default: <stem>_annotated.jpg)")
    ap.add_argument("--json", type=Path, help="bbox json (default: <stem>_bboxes.json)")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"cannot read {args.image}")

    boxes: list[dict] = []
    win = "annotate (1=extended rail, 2=folded side, u=undo, s=save, q=quit)"
    clone = img.copy()
    drawing = False
    ix = iy = 0
    cur_label = LABELS[0]

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drawing, ix, iy, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            clone = img.copy()
            draw_boxes(clone, boxes)
            cv2.rectangle(clone, (ix, iy), (x, y), COLORS[cur_label], 2)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x0, x1 = sorted((ix, x))
            y0, y1 = sorted((iy, y))
            if x1 - x0 > 4 and y1 - y0 > 4:
                boxes.append({"label": cur_label, "bbox": [x0, y0, x1, y1]})
            clone = img.copy()
            draw_boxes(clone, boxes)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, clone)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("1"):
            cur_label = LABELS[0]
            print("label:", cur_label)
        if key == ord("2"):
            cur_label = LABELS[1]
            print("label:", cur_label)
        if key == ord("u") and boxes:
            boxes.pop()
            clone = img.copy()
            draw_boxes(clone, boxes)
        if key == ord("s"):
            break

    cv2.destroyAllWindows()

    out_img = args.out or args.image.with_name(f"{args.image.stem}_annotated.jpg")
    out_json = args.json or args.image.with_name(f"{args.image.stem}_bboxes.json")
    annotated = img.copy()
    draw_boxes(annotated, boxes)
    cv2.imwrite(str(out_img), annotated)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"image": str(args.image), "boxes": boxes}, f, indent=2, ensure_ascii=False)
    print("saved", out_img)
    print("saved", out_json)


if __name__ == "__main__":
    main()
