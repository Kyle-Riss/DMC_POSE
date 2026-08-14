#!/usr/bin/env python3
"""Generate review-only temporal boundary proposals for FallVision clips."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

BOUNDARIES = ("fall_onset_frame", "impact_frame", "post_fall_stable_frame", "fall_end_frame")


@dataclass(frozen=True)
class Params:
    smooth_seconds: float
    onset_peak_ratio: float
    onset_sustain_seconds: float
    stable_peak_ratio: float
    stable_sustain_seconds: float
    pose_weight: float = 0.0


def moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) < 2:
        return values.astype(np.float64, copy=True)
    size = 2 * radius + 1
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(size) / size, mode="valid")


def extract_motion_curve(video_path: Path, target_width: int = 320) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid fps for {video_path}: {fps}")
    previous = None
    scores: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if width > target_width:
            frame = cv2.resize(frame, (target_width, max(1, round(height * target_width / width))), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if previous is None:
            scores.append(0.0)
        else:
            delta = cv2.absdiff(gray, previous)
            scores.append(float(np.mean(delta >= 10)) + 0.35 * float(np.mean(delta)) / 255.0)
        previous = gray
    capture.release()
    if len(scores) < 2:
        raise RuntimeError(f"too few decoded frames: {video_path}")
    return np.asarray(scores, dtype=np.float64), fps


def _robust_unit(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(values, 95)) if len(values) else 0.0
    if scale <= 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values / scale, 0.0, 3.0)


def extract_pose_motion_curve(video_path: Path, pose_model, device: str, frame_width: int = 640) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    states: list[tuple[float, float, float] | None] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if width != frame_width:
            frame = cv2.resize(
                frame,
                (frame_width, max(1, round(height * frame_width / width))),
                interpolation=cv2.INTER_AREA,
            )
        result_list = pose_model.predict(frame, device=device, verbose=False)
        result = result_list[0] if result_list else None
        if result is None or result.keypoints is None or len(result.keypoints) == 0:
            states.append(None)
            continue
        xy_all = result.keypoints.xy.cpu().numpy()
        conf_all = (
            result.keypoints.conf.cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones(xy_all.shape[:2], dtype=np.float32)
        )
        candidates = []
        for xy, conf in zip(xy_all, conf_all):
            visible = (conf >= 0.20) & (xy[:, 0] > 0) & (xy[:, 1] > 0)
            if int(visible.sum()) < 5:
                continue
            points = xy[visible]
            area = max(1.0, float(np.ptp(points[:, 0]) * np.ptp(points[:, 1])))
            candidates.append((area * float(np.mean(conf[visible])), xy, conf, visible))
        if not candidates:
            states.append(None)
            continue
        _, xy, conf, visible = max(candidates, key=lambda item: item[0])
        center = (
            (xy[11] + xy[12]) / 2.0
            if conf[11] >= 0.20 and conf[12] >= 0.20
            else np.mean(xy[visible], axis=0)
        )
        points = xy[visible]
        bbox_w = max(1.0, float(np.ptp(points[:, 0])))
        bbox_h = max(1.0, float(np.ptp(points[:, 1])))
        torso = 0.0
        if all(conf[index] >= 0.20 for index in (5, 6, 11, 12)):
            shoulder = (xy[5] + xy[6]) / 2.0
            hip = (xy[11] + xy[12]) / 2.0
            torso = float(
                math.atan2(abs(shoulder[0] - hip[0]), abs(shoulder[1] - hip[1]) + 1e-6)
                / (math.pi / 2.0)
            )
        states.append(
            (
                float(center[0] / frame.shape[1]),
                float(center[1] / frame.shape[0]),
                float(torso + 0.25 * bbox_w / bbox_h),
            )
        )
    capture.release()
    if not states:
        return np.zeros(0, dtype=np.float64)
    array = np.full((len(states), 3), np.nan, dtype=np.float64)
    for index, state in enumerate(states):
        if state is not None:
            array[index] = state
    for column in range(array.shape[1]):
        valid = np.flatnonzero(np.isfinite(array[:, column]))
        if len(valid):
            array[:, column] = np.interp(np.arange(len(array)), valid, array[valid, column])
        else:
            array[:, column] = 0.0
    delta_xy = np.linalg.norm(np.diff(array[:, :2], axis=0, prepend=array[:1, :2]), axis=1)
    delta_shape = np.abs(np.diff(array[:, 2], prepend=array[0, 2]))
    return _robust_unit(delta_xy) + 0.45 * _robust_unit(delta_shape)


def extract_provided_keypoint_curve(csv_path: Path) -> np.ndarray:
    frames: dict[int, dict[str, tuple[float, float, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["Frame"]) - 1
            frames.setdefault(frame, {})[row["Keypoint"].strip().lower()] = (
                float(row["X"]), float(row["Y"]), float(row["Confidence"])
            )
    if not frames:
        return np.zeros(0, dtype=np.float64)
    names = ("left shoulder", "right shoulder", "left hip", "right hip")
    states = np.full((max(frames) + 1, 4), np.nan, dtype=np.float64)
    for frame_index, points_by_name in frames.items():
        visible = [point for point in points_by_name.values() if point[2] >= 0.20]
        if len(visible) < 5:
            continue
        xy = np.asarray([[point[0], point[1]] for point in visible])
        bbox_w = max(1.0, float(np.ptp(xy[:, 0])))
        bbox_h = max(1.0, float(np.ptp(xy[:, 1])))
        left_hip, right_hip = points_by_name.get(names[2]), points_by_name.get(names[3])
        if left_hip and right_hip and left_hip[2] >= 0.20 and right_hip[2] >= 0.20:
            center_x = (left_hip[0] + right_hip[0]) / 2.0
            center_y = (left_hip[1] + right_hip[1]) / 2.0
        else:
            center_x, center_y = np.mean(xy, axis=0)
        torso = 0.0
        named = [points_by_name.get(name) for name in names]
        if all(point is not None and point[2] >= 0.20 for point in named):
            shoulder_x = (named[0][0] + named[1][0]) / 2.0
            shoulder_y = (named[0][1] + named[1][1]) / 2.0
            hip_x = (named[2][0] + named[3][0]) / 2.0
            hip_y = (named[2][1] + named[3][1]) / 2.0
            torso = math.atan2(abs(shoulder_x - hip_x), abs(shoulder_y - hip_y) + 1e-6) / (math.pi / 2.0)
        states[frame_index] = (center_x, center_y, torso + 0.25 * bbox_w / bbox_h, math.hypot(bbox_w, bbox_h))
    for column in range(states.shape[1]):
        valid = np.flatnonzero(np.isfinite(states[:, column]))
        if len(valid):
            states[:, column] = np.interp(np.arange(len(states)), valid, states[valid, column])
        else:
            states[:, column] = 0.0
    person_scale = max(1.0, float(np.median(states[:, 3])))
    delta_xy = np.linalg.norm(np.diff(states[:, :2], axis=0, prepend=states[:1, :2]), axis=1) / person_scale
    delta_shape = np.abs(np.diff(states[:, 2], prepend=states[0, 2]))
    return _robust_unit(delta_xy) + 0.45 * _robust_unit(delta_shape)


def _first_sustained(mask: np.ndarray, start: int, stop: int, count: int) -> int | None:
    count = max(1, count)
    stop = min(stop, len(mask) - count + 1)
    for index in range(max(0, start), max(0, stop)):
        if bool(np.all(mask[index:index + count])):
            return index
    return None


def propose(curves: dict[str, np.ndarray], fps: float, params: Params) -> dict[str, int | float]:
    motion = _robust_unit(curves["motion"])
    pose = curves.get("pose")
    signal = motion if pose is None else motion + params.pose_weight * pose
    score = moving_average(signal, max(0, round(params.smooth_seconds * fps)))
    early_count = max(3, min(len(score) // 5, round(0.5 * fps)))
    baseline = float(np.median(score[:early_count]))
    impact = int(np.argmax(score))
    peak = float(score[impact])
    amplitude = max(peak - baseline, 1e-9)
    onset_threshold = baseline + params.onset_peak_ratio * amplitude
    onset = _first_sustained(score >= onset_threshold, 0, impact + 1, max(1, round(params.onset_sustain_seconds * fps)))
    if onset is None:
        onset = impact
    shoulder = baseline + 0.5 * params.onset_peak_ratio * amplitude
    while onset > 0 and score[onset - 1] >= shoulder:
        onset -= 1
    stable_threshold = baseline + params.stable_peak_ratio * amplitude
    stable = _first_sustained(score <= stable_threshold, impact + 1, len(score), max(1, round(params.stable_sustain_seconds * fps)))
    if stable is None:
        stable = len(score) - 1
    uncertainty = max(1, round(0.15 * fps))
    return {
        "proposed_fall_onset_frame": int(onset),
        "proposed_impact_frame": impact,
        "proposed_post_fall_stable_frame": int(max(impact, stable)),
        "proposed_fall_end_frame": len(score) - 1,
        "proposed_onset_earliest_frame": int(max(0, onset - uncertainty)),
        "proposed_onset_latest_frame": int(min(impact, onset + uncertainty)),
        "motion_baseline": baseline, "motion_peak": peak, "motion_amplitude": amplitude,
    }


def parameter_grid() -> Iterable[Params]:
    for smooth in (0.03, 0.06, 0.10, 0.15):
        for onset_ratio in (0.08, 0.12, 0.18, 0.25, 0.35):
            for onset_sustain in (0.03, 0.06, 0.10):
                for stable_ratio in (0.08, 0.12, 0.18):
                    for pose_weight in (0.0, 0.25, 0.5, 1.0):
                        yield Params(smooth, onset_ratio, onset_sustain, stable_ratio, 0.30, pose_weight)


def boundary_loss(rows: list[dict], curves: dict[str, tuple[dict[str, np.ndarray], float]], params: Params) -> float:
    errors = []
    for row in rows:
        prediction = propose(*curves[row["video_id"]], params)
        for name in ("fall_onset_frame", "impact_frame"):
            errors.append(abs(int(prediction[f"proposed_{name}"]) - int(row[name])) / float(row["fps"]))
    return float(np.mean(errors)) if errors else float("inf")


def select_params(rows: list[dict], curves: dict[str, tuple[dict[str, np.ndarray], float]]) -> Params:
    return min(parameter_grid(), key=lambda candidate: boundary_loss(rows, curves, candidate))


def summarize(rows: list[dict], proposals: list[dict]) -> dict:
    by_id = {row["video_id"]: row for row in rows}
    metrics = {}
    for boundary in BOUNDARIES:
        frame_errors, second_errors = [], []
        for proposal_row in proposals:
            source = by_id[proposal_row["video_id"]]
            if source.get(boundary, "") == "":
                continue
            error = abs(int(proposal_row[f"proposed_{boundary}"]) - int(source[boundary]))
            frame_errors.append(error)
            second_errors.append(error / float(source["fps"]))
        metrics[boundary] = {
            "count": len(frame_errors),
            "mae_frames": float(np.mean(frame_errors)) if frame_errors else None,
            "median_frames": float(np.median(frame_errors)) if frame_errors else None,
            "mae_seconds": float(np.mean(second_errors)) if second_errors else None,
            "median_seconds": float(np.median(second_errors)) if second_errors else None,
        }
    return metrics


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=project / "external_datasets/annotations/fallvision_pilot_v1_complete.csv")
    parser.add_argument("--output", type=Path, default=project / "external_datasets/annotations/fallvision_pilot_v1_proposals.csv")
    parser.add_argument("--report", type=Path, default=project / "external_datasets/annotations/fallvision_pilot_v1_proposals_report.json")
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--calibration-annotations", type=Path)
    parser.add_argument("--pose-model", type=Path, default=project / "yolo11m-pose.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--provided-only", action="store_true")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=project / "external_datasets/manifests/fallvision_canonical_inventory_v1.csv",
    )
    parser.add_argument(
        "--provided-keypoints-root",
        type=Path,
        default=project / "external_datasets/fallvision/provided_keypoints",
    )
    args = parser.parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as handle:
        target_rows = list(csv.DictReader(handle))
    calibration_path = args.calibration_annotations or args.annotations
    with calibration_path.open(newline="", encoding="utf-8") as handle:
        calibration_rows = [
            row for row in csv.DictReader(handle)
            if row.get("annotation_status") == "complete"
        ]
    rows_by_id = {row["video_id"]: row for row in target_rows}
    rows_by_id.update({row["video_id"]: row for row in calibration_rows})
    rows = list(rows_by_id.values())
    pose_model = None
    if not args.skip_pose and not args.provided_only:
        from ultralytics import YOLO
        pose_model = YOLO(str(args.pose_model))
    inventory = {}
    if args.provided_only:
        with args.inventory.open(newline="", encoding="utf-8") as handle:
            inventory = {row["canonical_id"]: row for row in csv.DictReader(handle)}
    curves: dict[str, tuple[dict[str, np.ndarray], float]] = {}
    failures = []
    for index, row in enumerate(rows, 1):
        try:
            if args.provided_only:
                item = inventory[row["video_id"]]
                keypoint_path = args.provided_keypoints_root / item["keypoint_csv_member"]
                components = {"motion": extract_provided_keypoint_curve(keypoint_path)}
                fps = float(row["fps"])
            else:
                motion, fps = extract_motion_curve(Path(row["local_video_path"]), args.target_width)
                components = {"motion": motion}
                if pose_model is not None:
                    pose = extract_pose_motion_curve(Path(row["local_video_path"]), pose_model, args.device)
                    if len(pose) == len(motion):
                        components["pose"] = pose
            curves[row["video_id"]] = (components, fps)
            print(f"[{index:02d}/{len(rows):02d}] signal {row['recording_id']}", flush=True)
        except Exception as error:
            failures.append({"video_id": row["video_id"], "error": str(error)})
    usable_targets = [row for row in target_rows if row["video_id"] in curves]
    usable_calibration = [row for row in calibration_rows if row["video_id"] in curves]
    if not usable_targets:
        raise SystemExit("no usable videos")
    if not usable_calibration:
        raise SystemExit("no complete calibration annotations")
    selected = select_params(usable_calibration, curves)
    proposals = []
    for row in usable_targets:
        method = "provided_keypoints_v1" if args.provided_only else "frame_motion_pose_v2"
        proposals.append({"video_id": row["video_id"], "recording_id": row["recording_id"], "scene_id": row["scene_id"],
                          **propose(*curves[row["video_id"]], selected), "proposal_method": method,
                          "proposal_status": "review_required"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(proposals[0])); writer.writeheader(); writer.writerows(proposals)
    loo_proposals = []
    if args.calibration_annotations is None and len(usable_calibration) > 2:
        for held_out in usable_calibration:
            params = select_params([row for row in usable_calibration if row is not held_out], curves)
            loo_proposals.append({"video_id": held_out["video_id"], **propose(*curves[held_out["video_id"]], params)})
    report = {
        "schema_version": "fallvision_temporal_proposals_v1", "purpose": "review-only proposals; never ground truth",
        "source_annotations": str(args.annotations.resolve()), "output": str(args.output.resolve()),
        "video_count": len(usable_targets), "calibration_video_count": len(usable_calibration),
        "calibration_annotations": str(calibration_path.resolve()),
        "failures": failures, "selected_parameters": asdict(selected),
        "in_sample_metrics": summarize(usable_targets, proposals),
        "leave_one_video_out_metrics": summarize(usable_calibration, loo_proposals) if loo_proposals else None,
        "limitations": ["24-video pilot is too small for a generalization claim",
                        "global frame motion can react to helpers, bedding, or camera motion",
                        "all proposals require human review"],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
