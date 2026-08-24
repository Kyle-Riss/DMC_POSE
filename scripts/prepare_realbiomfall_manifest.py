#!/usr/bin/env python3
"""Build a fail-closed, train-augmentation-only RealBiomFall manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ARCHIVE_MD5 = {
    "labels-100.zip": "65a7c3b8346e9cff8497b5bdd5c80372",
    "video_clips-trimmed_cropped_padded_resized-100.zip": "1168284537004b7937ec552015461719",
}

_VIDEO_NAME = re.compile(
    r"^(?P<source>.+)_(?P<source_start>-?\d+(?:\.\d+)?)_"
    r"(?P<source_end>-?\d+(?:\.\d+)?)\.mp4$"
)


class RestrictedRealBiomUnpickler(pickle.Unpickler):
    """Allow only the two NumPy constructors present in the verified labels."""

    def find_class(self, module: str, name: str):
        if (module, name) == ("numpy.core.multiarray", "scalar"):
            return np.core.multiarray.scalar
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - dataset integrity, not cryptography
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives(raw_dir: Path) -> dict[str, str]:
    actual = {}
    for filename, expected in ARCHIVE_MD5.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        value = md5sum(path)
        if value != expected:
            raise ValueError(f"archive MD5 mismatch: {path}: {value} != {expected}")
        actual[filename] = value
    return actual


def load_verified_pickle(path: Path):
    with path.open("rb") as handle:
        return RestrictedRealBiomUnpickler(handle).load()


def annotation_time(value) -> float | None:
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        return float(value)
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
        and isinstance(value[0].get("t"), (int, float, np.number))
    ):
        return float(value[0]["t"])
    return None


def parse_video_name(filename: str) -> tuple[str, float, float]:
    match = _VIDEO_NAME.fullmatch(filename)
    if not match:
        raise ValueError(f"unexpected RealBiomFall filename: {filename}")
    return (
        match.group("source"),
        float(match.group("source_start")),
        float(match.group("source_end")),
    )


def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"readable": False, "fps": None, "frame_count": None, "duration_sec": None}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "readable": fps > 0.0 and frames > 0,
        "fps": round(fps, 6) if fps > 0.0 else None,
        "frame_count": frames,
        "duration_sec": round(frames / fps, 6) if fps > 0.0 else None,
        "width": width,
        "height": height,
    }


def build_manifest(labels_dir: Path, videos_dir: Path) -> dict:
    semantic = load_verified_pickle(labels_dir / "labels_semantical.pkl")
    coarse = load_verified_pickle(labels_dir / "labels_temporal_coarse.pkl")
    fine = load_verified_pickle(labels_dir / "labels_temporal_finegrained.pkl")
    key_sets = (set(semantic), set(coarse), set(fine))
    if not (key_sets[0] == key_sets[1] == key_sets[2]):
        raise ValueError("semantic/coarse/fine label keys do not match")

    items = []
    errors = []
    official_by_source: dict[str, Counter] = defaultdict(Counter)
    for filename in sorted(semantic):
        source_id, source_start, source_end = parse_video_name(filename)
        video_path = videos_dir / filename
        meta = video_metadata(video_path)
        sem = semantic[filename]
        onset = annotation_time(sem.get("start"))
        impact = annotation_time(sem.get("reaching lowest position"))
        annotated_end = annotation_time(sem.get("end of fall"))
        duration = meta.get("duration_sec")
        official_subset = str(coarse[filename].get("subset") or "unknown")
        official_by_source[source_id][official_subset] += 1

        excluded = ["subject_identity_unknown"]
        eligible = bool(meta.get("readable") and duration is not None and onset is not None)
        if not meta.get("readable"):
            excluded.append("unreadable_video")
            errors.append(f"unreadable video: {video_path}")
        if onset is None:
            excluded.append("missing_or_ambiguous_fall_start")
            errors.append(f"missing onset: {filename}")
        if eligible and not (0.0 <= onset <= duration):
            eligible = False
            excluded.append("fall_start_outside_video")
            errors.append(f"onset outside video: {filename}: {onset}/{duration}")
        if impact is not None and onset is not None and impact < onset:
            eligible = False
            excluded.append("impact_before_fall_start")
            errors.append(f"impact before onset: {filename}: {impact}/{onset}")

        intervals = []
        if eligible:
            onset = min(float(onset), float(duration))
            if onset > 0.0:
                intervals.append({"source_label": "pre_fall", "label": "non_fall", "start_sec": 0.0, "end_sec": round(onset, 6)})
            intervals.append({"source_label": "fall_and_post_fall", "label": "fall", "start_sec": round(onset, 6), "end_sec": round(float(duration), 6)})

        items.append(
            {
                "video_id": "realbiomfall_" + Path(filename).stem,
                "dataset": "realbiomfall_100_v1",
                "subject_id": None,
                "split": "train",
                "source_group": source_id,
                "split_group": "realbiomfall_source_video:" + source_id,
                "official_subset": official_subset,
                "official_subset_usable_for_promotion": False,
                "source_path": str(video_path.resolve()),
                "video_path": str(video_path.resolve()),
                "source_clip_start_sec": source_start,
                "source_clip_end_sec": source_end,
                "activity_label": "fall",
                "binary_fall_label": 1,
                "fall_start_sec": round(float(onset), 6) if onset is not None else None,
                "impact_sec": round(float(impact), 6) if impact is not None else None,
                "annotated_end_of_fall_sec": round(float(annotated_end), 6) if annotated_end is not None else None,
                "fall_end_sec": round(float(duration), 6) if eligible else None,
                "staged_or_real": "mixed_web_source",
                "annotation_source": str((labels_dir / "labels_semantical.pkl").resolve()),
                "annotation_scope": "original_temporal_and_semantic",
                "intervals": intervals,
                "training_eligible": eligible,
                "temporal_gru_eligible": eligible,
                "augmentation_train_only": True,
                "promotion_metric_eligible": False,
                "diagnostic_eligible": bool(meta.get("readable")),
                "precontext_sec": round(float(onset), 6) if onset is not None else None,
                "has_4s_precontext": bool(onset is not None and onset >= 4.0),
                "excluded_reasons": excluded,
                **meta,
            }
        )

    leaking_sources = sorted(
        source for source, counts in official_by_source.items() if len(counts) > 1
    )
    return {
        "schema_version": "temporal_manifest_v2",
        "dataset": "realbiomfall_100_v1",
        "license": "CC BY 4.0",
        "sample_hz_target": 20.0,
        "task": "binary temporal fall detection augmentation",
        "split_policy": "train_augmentation_only; official split rejected because source videos cross subsets",
        "promotion_metric_eligible": False,
        "warnings": [
            "positive-only dataset; do not train or evaluate as a standalone corpus",
            "subject identity is unavailable",
            "official training/testing subsets mix clips from the same source videos",
        ],
        "video_count": len(items),
        "duration_total_sec": round(sum(float(item.get("duration_sec") or 0.0) for item in items), 6),
        "training_eligible_count": sum(bool(item["training_eligible"]) for item in items),
        "onset_count": sum(item["fall_start_sec"] is not None for item in items),
        "impact_count": sum(item["impact_sec"] is not None for item in items),
        "four_second_precontext_count": sum(item["has_4s_precontext"] for item in items),
        "source_group_counts": dict(sorted(Counter(item["source_group"] for item in items).items())),
        "official_subset_counts": dict(sorted(Counter(item["official_subset"] for item in items).items())),
        "official_split_leaking_source_groups": leaking_sources,
        "errors": errors,
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    extracted = project / "external_datasets/realbiomfall/extracted"
    parser.add_argument("--labels-dir", type=Path, default=extracted / "labels-100")
    parser.add_argument("--videos-dir", type=Path, default=extracted / "video_clips-trimmed_cropped_padded_resized-100")
    parser.add_argument("--raw-dir", type=Path, default=project / "external_datasets/realbiomfall/raw")
    parser.add_argument("--out", type=Path, default=project / "external_datasets/manifests/realbiomfall_100_train_augmentation_v1.json")
    parser.add_argument("--skip-archive-verification", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    archive_md5 = {} if args.skip_archive_verification else verify_archives(args.raw_dir)
    payload = build_manifest(args.labels_dir.resolve(), args.videos_dir.resolve())
    payload["archive_md5"] = archive_md5
    payload["labels_dir"] = str(args.labels_dir.resolve())
    payload["videos_dir"] = str(args.videos_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_keys = (
        "video_count", "duration_total_sec", "training_eligible_count", "onset_count",
        "impact_count", "four_second_precontext_count", "source_group_counts",
        "official_subset_counts", "official_split_leaking_source_groups", "errors",
    )
    print(json.dumps({key: payload[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    print(f"manifest: {args.out.resolve()}")
    return 2 if args.strict and payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
