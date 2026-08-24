#!/usr/bin/env python3
"""Evaluate causal TCN outputs as contiguous fall events, not frame windows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

from temporal_model import architecture_from_checkpoint, build_temporal_model
from train_tcn import probabilities


def contiguous_events(rows: list[dict], probability: np.ndarray, threshold: float, persistence: int, merge_gap_sec: float = 0.0) -> list[dict]:
    raw = probability >= threshold
    confirmed = np.zeros_like(raw, dtype=bool)
    boundary = np.ones(len(rows), dtype=bool)
    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        boundary[index] = (
            current.get("sequence_id", 1) != previous.get("sequence_id", 1)
            or current.get("track_id", 1) != previous.get("track_id", 1)
        )

    run = 0
    for index, active in enumerate(raw):
        if boundary[index]:
            run = 0
        run = run + 1 if active else 0
        if run >= persistence:
            confirmed[index - persistence + 1 : index + 1] = True

    events = []
    start = None
    for index, active in enumerate(confirmed):
        if boundary[index] and start is not None:
            end = index - 1
            events.append({
                "start_sec": float(rows[start]["end_sec"]),
                "end_sec": float(rows[end]["end_sec"]),
                "peak_probability": round(float(probability[start : end + 1].max()), 6),
                "sequence_id": rows[start].get("sequence_id", 1),
            })
            start = None
        if active and start is None:
            start = index
        if start is not None and not active:
            end = index - 1
            events.append({
                "start_sec": float(rows[start]["end_sec"]),
                "end_sec": float(rows[end]["end_sec"]),
                "peak_probability": round(float(probability[start : end + 1].max()), 6),
                "sequence_id": rows[start].get("sequence_id", 1),
            })
            start = None
    if start is not None:
        end = len(confirmed) - 1
        events.append({
            "start_sec": float(rows[start]["end_sec"]),
            "end_sec": float(rows[end]["end_sec"]),
            "peak_probability": round(float(probability[start : end + 1].max()), 6),
            "sequence_id": rows[start].get("sequence_id", 1),
        })

    if merge_gap_sec <= 0 or len(events) < 2:
        return events
    merged = [events[0].copy()]
    for event in events[1:]:
        previous = merged[-1]
        same_sequence = event["sequence_id"] == previous["sequence_id"]
        if same_sequence and event["start_sec"] - previous["end_sec"] <= merge_gap_sec:
            previous["end_sec"] = event["end_sec"]
            previous["peak_probability"] = max(previous["peak_probability"], event["peak_probability"])
        else:
            merged.append(event.copy())
    return merged

def overlap(a0, a1, b0, b1):
    return max(a0, b0) <= min(a1, b1)


def poisson_rate_ci(count: int, exposure_hours: float, confidence: float = 0.95) -> list[float] | None:
    if exposure_hours <= 0:
        return None
    alpha = 1.0 - confidence
    lower_count = 0.0 if count == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * count)
    upper_count = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (count + 1))
    return [round(float(lower_count / exposure_hours), 2), round(float(upper_count / exposure_hours), 2)]


def event_is_pre_onset_ready(event: dict, rows: list[dict], max_interval_sec: float = 0.150) -> bool:
    onset = float(event["start_sec"])
    seen = set()
    for row in rows:
        key = (row.get("track_id", 1), row.get("sequence_id", 1))
        if key in seen:
            continue
        seen.add(key)
        ready_sec = row.get("sequence_ready_sec")
        observation_end = row.get("sequence_observation_end_sec")
        if ready_sec is None or observation_end is None:
            continue
        if float(ready_sec) <= onset and float(observation_end) >= onset - max_interval_sec:
            return True
    return False


def event_has_persistence_capacity(event: dict, rows: list[dict], persistence: int) -> bool:
    """Return whether GT contains enough same-sequence windows to confirm it.

    This is probability-independent.  It distinguishes a classifier miss from
    an event that can never satisfy the requested persistence because the
    observed-only sequence is too short.
    """
    required = max(1, int(persistence))
    run = 0
    previous_key = None
    start_sec = float(event["start_sec"])
    end_sec = float(event["end_sec"])
    for row in rows:
        key = (row.get("track_id", 1), row.get("sequence_id", 1))
        inside = start_sec <= float(row["end_sec"]) <= end_sec
        if key != previous_key or not inside:
            run = 0
        if inside:
            run += 1
            if run >= required:
                return True
        previous_key = key
    return False


def predict_split(windows_dir: Path, split: str, checkpoint_path: Path, device: torch.device) -> tuple[list[dict], np.ndarray]:
    data = np.load(windows_dir / f"{split}.npz")
    x = data["x"].astype(np.float32)
    metadata = json.loads((windows_dir / f"{split}_metadata.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    x = (x - checkpoint["mean"]) / checkpoint["std"]
    model = build_temporal_model(architecture_from_checkpoint(checkpoint), int(checkpoint["feature_count"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    probability = probabilities(model, x, device)
    return metadata, probability


def evaluate_split(split: str, windows_dir: Path, manifest: dict, checkpoint_path: Path, threshold: float, persistence: int, device: torch.device, merge_gap_sec: float = 0.0, *, precomputed_metadata: list[dict] | None = None, precomputed_probability: np.ndarray | None = None) -> dict:
    if precomputed_metadata is None or precomputed_probability is None:
        metadata, probability = predict_split(windows_dir, split, checkpoint_path, device)
    else:
        metadata = precomputed_metadata
        probability = np.asarray(precomputed_probability, dtype=np.float32)
    if len(metadata) != len(probability):
        raise ValueError("metadata/probability length mismatch")

    manifest_by_id = {item["video_id"]: item for item in manifest["items"] if item["split"] == split}
    grouped = defaultdict(list)
    grouped_probs = defaultdict(list)
    for row, prob in zip(metadata, probability):
        grouped[row["video_id"]].append(row)
        grouped_probs[row["video_id"]].append(float(prob))

    gt_total = pre_onset_ready_total = evaluable_gt_total = persistence_evaluable_total = detected = false_events = predicted_total = 0
    onset_latencies = []
    impact_latencies = []
    per_video = []
    total_duration = 0.0
    for video_id, item in manifest_by_id.items():
        duration = float(item.get("duration_sec") or item.get("declared_duration_sec") or 0.0)
        total_duration += duration
        gt = [interval for interval in item.get("intervals", []) if interval["label"] == "fall"]
        gt_total += len(gt)
        rows = grouped.get(video_id, [])
        video_evaluable = sum(
            any(float(event["start_sec"]) <= float(row["end_sec"]) <= float(event["end_sec"]) for row in rows)
            for event in gt
        )
        video_pre_onset_ready = sum(event_is_pre_onset_ready(event, rows) for event in gt)
        video_persistence_evaluable = sum(
            event_has_persistence_capacity(event, rows, persistence) for event in gt
        )
        evaluable_gt_total += video_evaluable
        persistence_evaluable_total += video_persistence_evaluable
        pre_onset_ready_total += video_pre_onset_ready
        probs = np.asarray(grouped_probs.get(video_id, []), dtype=np.float32)
        pred = contiguous_events(rows, probs, threshold, persistence, merge_gap_sec) if rows else []
        predicted_total += len(pred)
        matched_pred = set()
        matched_gt = 0
        for gt_event in gt:
            matches = [
                (index, event)
                for index, event in enumerate(pred)
                if index not in matched_pred and overlap(float(gt_event["start_sec"]), float(gt_event["end_sec"]), event["start_sec"], event["end_sec"])
            ]
            if matches:
                index, event = min(matches, key=lambda pair: pair[1]["start_sec"])
                matched_pred.add(index)
                matched_gt += 1
                onset_latencies.append(event["start_sec"] - float(gt_event["start_sec"]))
                impact_sec = gt_event.get("impact_sec", item.get("impact_sec"))
                if impact_sec is not None:
                    impact_latencies.append(event["start_sec"] - float(impact_sec))
        detected += matched_gt
        video_false = len(pred) - len(matched_pred)
        false_events += video_false
        per_video.append({
            "video_id": video_id,
            "gt_events": len(gt),
            "pre_onset_ready_events": video_pre_onset_ready,
            "evaluable_gt_events": video_evaluable,
            "persistence_evaluable_gt_events": video_persistence_evaluable,
            "predicted_events": len(pred),
            "matched_gt": matched_gt,
            "false_events": video_false,
            "events": pred,
        })

    precision = detected / predicted_total if predicted_total else 0.0
    recall = detected / gt_total if gt_total else 0.0
    duration_hours = total_duration / 3600.0
    return {
        "split": split,
        "threshold": threshold,
        "persistence_windows": persistence,
        "merge_gap_sec": merge_gap_sec,
        "videos": len(manifest_by_id),
        "duration_hours": round(duration_hours, 6),
        "gt_events": gt_total,
        "pre_onset_ready_events": pre_onset_ready_total,
        "pre_onset_ready_coverage": round(pre_onset_ready_total / gt_total, 4) if gt_total else None,
        "evaluable_gt_events": evaluable_gt_total,
        "event_evaluable_coverage": round(evaluable_gt_total / gt_total, 4) if gt_total else None,
        "persistence_evaluable_gt_events": persistence_evaluable_total,
        "persistence_evaluable_coverage": round(persistence_evaluable_total / gt_total, 4) if gt_total else None,
        "videos_with_windows": len(grouped),
        "predicted_events": predicted_total,
        "detected_events": detected,
        "false_events": false_events,
        "event_precision": round(precision, 4),
        "end_to_end_event_recall": round(recall, 4),
        "conditional_event_recall": round(detected / evaluable_gt_total, 4) if evaluable_gt_total else None,
        "false_events_per_hour": round(false_events / duration_hours, 2) if duration_hours else None,
        "false_events_per_hour_poisson_95ci": poisson_rate_ci(false_events, duration_hours),
        "latency_from_onset_sec_median": round(float(np.median(onset_latencies)), 3) if onset_latencies else None,
        "latency_from_onset_sec_p90": round(float(np.percentile(onset_latencies, 90)), 3) if onset_latencies else None,
        "latency_from_impact_sec_median": round(float(np.median(impact_latencies)), 3) if impact_latencies else None,
        "latency_from_impact_sec_p90": round(float(np.percentile(impact_latencies, 90)), 3) if impact_latencies else None,
        "per_video": per_video,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--windows-dir", type=Path, default=project_root / "external_datasets/windows/tcn_109_v2_no_missing/gmdcsa24_3s")
    parser.add_argument("--manifest", type=Path, default=project_root / "external_datasets/manifests/gmdcsa24.json")
    parser.add_argument("--model-dir", type=Path, default=project_root / "runs/temporal_tcn/gmdcsa24_tcn_v2_observed_only")
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--merge-gap-sec", type=float, default=3.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--split", action="append", choices=("train", "val", "test"),
        help="split to evaluate; repeat as needed (default: val and test)",
    )
    parser.add_argument(
        "--out", type=Path,
        help="output JSON path (default: MODEL_DIR/event_report.json)",
    )
    args = parser.parse_args()

    train_report = json.loads((args.model_dir / "report.json").read_text(encoding="utf-8"))
    threshold = float(train_report["validation"]["threshold"])
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    splits = args.split or ["val", "test"]
    reports = {
        split: evaluate_split(
            split, args.windows_dir, manifest, args.model_dir / "model.pt",
            threshold, args.persistence, device, args.merge_gap_sec,
        )
        for split in splits
    }
    output = args.out or (args.model_dir / "event_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {split: {key: value for key, value in report.items() if key != "per_video"} for split, report in reports.items()}
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    print(f"event_report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
