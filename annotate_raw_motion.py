#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def build_segment_index(segments: list[dict]) -> dict[str, list[tuple[int, float, float]]]:
    by_video: dict[str, list[tuple[int, float, float]]] = {}
    for i, seg in enumerate(segments):
        by_video.setdefault(seg["video_path"], []).append(
            (i, float(seg["start_sec"]), float(seg["end_sec"]))
        )
    for path, arr in by_video.items():
        arr.sort(key=lambda x: x[1])
    return by_video


def annotate_video_segments(
    video_path: str,
    segment_refs: list[tuple[int, float, float]],
    segments: list[dict],
    sample_fps: float,
    diff_threshold: int,
    min_blob_ratio: float,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        for seg_idx, _, _ in segment_refs:
            segments[seg_idx]["motion_error"] = "cannot_open_video"
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 20.0
    frame_interval = max(int(round(fps / max(sample_fps, 0.1))), 1)

    bins = []
    for seg_idx, start, end in segment_refs:
        bins.append(
            {
                "idx": seg_idx,
                "start": start,
                "end": end,
                "frames": 0,
                "motion_ratio_sum": 0.0,
                "body_like_frames": 0,
            }
        )

    current_bin = 0
    prev_gray = None
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % frame_interval != 0:
            frame_i += 1
            continue

        t_sec = frame_i / fps
        while current_bin < len(bins) and t_sec >= bins[current_bin]["end"]:
            current_bin += 1
        if current_bin >= len(bins):
            break
        if t_sec < bins[current_bin]["start"]:
            frame_i += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, motion_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            motion_pixels = int(np.count_nonzero(motion_mask))
            h, w = motion_mask.shape[:2]
            frame_area = float(h * w)
            motion_ratio = motion_pixels / frame_area if frame_area > 0 else 0.0

            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(motion_mask, 8)
            largest = 0
            for li in range(1, n_labels):
                largest = max(largest, int(stats[li, cv2.CC_STAT_AREA]))
            largest_ratio = largest / frame_area if frame_area > 0 else 0.0
            body_like = largest_ratio >= min_blob_ratio

            b = bins[current_bin]
            b["frames"] += 1
            b["motion_ratio_sum"] += motion_ratio
            if body_like:
                b["body_like_frames"] += 1

        prev_gray = gray
        frame_i += 1

    cap.release()

    # compute per-segment scores
    for b in bins:
        seg = segments[b["idx"]]
        frames = b["frames"]
        if frames == 0:
            seg["motion_score"] = 0.0
            seg["body_like_frame_ratio"] = 0.0
            seg["motion_sampled_frames"] = 0
            continue
        mean_motion = b["motion_ratio_sum"] / frames
        body_like_ratio = b["body_like_frames"] / frames
        # weight whole-body movement higher than partial local movement
        score = mean_motion * (0.4 + 1.6 * body_like_ratio)
        seg["motion_score"] = round(float(score), 8)
        seg["body_like_frame_ratio"] = round(float(body_like_ratio), 6)
        seg["motion_sampled_frames"] = int(frames)


def apply_video_thresholds(segments: list[dict], segment_indices: list[int]) -> None:
    scores = [float(segments[i].get("motion_score", 0.0)) for i in segment_indices]
    p75 = percentile(scores, 75)
    p90 = percentile(scores, 90)

    for i in segment_indices:
        s = segments[i]
        score = float(s.get("motion_score", 0.0))
        if score >= p90:
            level = "high_motion"
        elif score >= p75:
            level = "caution"
        else:
            level = "normal"
        s["motion_level"] = level
        s["motion_threshold_p75"] = round(p75, 8)
        s["motion_threshold_p90"] = round(p90, 8)


def apply_rotation_review_rule(
    segments: list[dict],
    segment_indices: list[int],
    min_body_like_ratio: float,
    min_consecutive_segments: int,
) -> None:
    # require whole-body-like movement AND consecutive high_motion segments
    run = 0
    for i in segment_indices:
        s = segments[i]
        high = s.get("motion_level") == "high_motion"
        body = float(s.get("body_like_frame_ratio", 0.0)) >= min_body_like_ratio
        if high and body:
            run += 1
        else:
            run = 0

        if run >= min_consecutive_segments:
            s["rotation_decision_status"] = "review_required_high_motion"
            if not s.get("notes"):
                s["notes"] = "high whole-body motion (consecutive segments)"
        else:
            if s.get("rotation_decision_status") != "locked_by_manual_review":
                s["rotation_decision_status"] = "pending_definition"


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate raw segment motion levels.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/dmc/Dataset/Raw_data/raw_segments_manifest_30s.json"),
    )
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--diff-threshold", type=int, default=20)
    parser.add_argument(
        "--min-blob-ratio",
        type=float,
        default=0.015,
        help="Connected-motion blob area ratio to consider body-like movement.",
    )
    parser.add_argument(
        "--review-body-like-ratio",
        type=float,
        default=0.15,
        help="Segment-level body_like_frame_ratio threshold for rotation review.",
    )
    parser.add_argument(
        "--review-consecutive",
        type=int,
        default=2,
        help="Required consecutive high-motion segments for review trigger.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments: list[dict] = data.get("segments", [])
    by_video = build_segment_index(segments)

    for video_path, refs in by_video.items():
        annotate_video_segments(
            video_path=video_path,
            segment_refs=refs,
            segments=segments,
            sample_fps=args.sample_fps,
            diff_threshold=args.diff_threshold,
            min_blob_ratio=args.min_blob_ratio,
        )

        seg_indices = [idx for idx, _, _ in refs]
        apply_video_thresholds(segments, seg_indices)
        apply_rotation_review_rule(
            segments,
            seg_indices,
            min_body_like_ratio=args.review_body_like_ratio,
            min_consecutive_segments=args.review_consecutive,
        )

    data["motion_annotation_config"] = {
        "sample_fps": args.sample_fps,
        "diff_threshold": args.diff_threshold,
        "min_blob_ratio": args.min_blob_ratio,
        "review_body_like_ratio": args.review_body_like_ratio,
        "review_consecutive": args.review_consecutive,
        "rule": "review_required_high_motion when high_motion + body_like_frame_ratio>=review_body_like_ratio for >=review_consecutive consecutive segments",
    }

    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)
    print(f"segments={len(segments)} videos={len(by_video)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
