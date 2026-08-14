#!/usr/bin/env python3
# Build a provenance-preserving FallVision raw/mask/keypoint inventory from RAR listings.

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ARCHIVE_RE = re.compile(
    r"^(?P<label>f|nf)_(?P<kind>raw|mask)_(?P<scene>b|c|s)_(?P<chunk>[0-9]+)"
    r"(?P<keypoints>_(?:keypoints|ketpoints)_csv)?[.]rar$",
    re.IGNORECASE,
)
SCENES = {"b": "bed", "c": "chair", "s": "stand"}


def parse_archive_name(name: str) -> dict | None:
    match = ARCHIVE_RE.match(name)
    if match is None:
        return None
    groups = match.groupdict()
    return {
        "activity_label": "fall" if groups["label"].lower() == "f" else "non_fall",
        "binary_fall_label": 1 if groups["label"].lower() == "f" else 0,
        "scene_id": SCENES[groups["scene"].lower()],
        "chunk_id": int(groups["chunk"]),
        "modality": "keypoint_csv" if groups["keypoints"] else groups["kind"].lower(),
    }


def recording_id_from_member(member: str, modality: str) -> str | None:
    name = Path(member).name
    suffix = Path(name).suffix.lower()
    expected = ".csv" if modality == "keypoint_csv" else ".mp4"
    if suffix != expected:
        return None
    stem = Path(name).stem
    if modality == "keypoint_csv" and stem.lower().endswith("_keypoints"):
        stem = stem[: -len("_keypoints")]
    return stem or None


def determine_pair_status(raw: bool, mask: bool, keypoint_csv: bool) -> str:
    if raw and mask and keypoint_csv:
        return "complete"
    if raw and mask:
        return "missing_csv"
    if raw and keypoint_csv:
        return "missing_mask"
    if mask and keypoint_csv:
        return "missing_raw"
    if raw:
        return "raw_only"
    if mask:
        return "mask_only"
    if keypoint_csv:
        return "csv_only"
    return "excluded"


def list_archive(unrar: Path, archive: Path) -> list[str]:
    process = subprocess.run(
        [str(unrar), "lb", str(archive)],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"unrar failed for {archive}: {process.stderr[-500:]}")
    return [line.strip() for line in process.stdout.splitlines() if line.strip()]


def load_aliases(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_inventory(files_root: Path, unrar: Path, aliases_path: Path) -> dict:
    records: dict[tuple[str, str, int, str], dict] = {}
    errors: list[str] = []
    archive_count = 0

    for archive in sorted(files_root.glob("*.rar")):
        spec = parse_archive_name(archive.name)
        if spec is None:
            errors.append(f"unrecognized_archive:{archive.name}")
            continue
        archive_count += 1
        try:
            members = list_archive(unrar, archive)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        modality = spec["modality"]
        for member in members:
            recording_id = recording_id_from_member(member, modality)
            if recording_id is None:
                continue
            key = (
                spec["activity_label"],
                spec["scene_id"],
                spec["chunk_id"],
                recording_id,
            )
            row = records.setdefault(
                key,
                {
                    "dataset": "fallvision",
                    "canonical_id": "fallvision:"
                    + ":".join(map(str, key)),
                    "recording_id": recording_id,
                    "activity_label": spec["activity_label"],
                    "binary_fall_label": spec["binary_fall_label"],
                    "scene_id": spec["scene_id"],
                    "chunk_id": spec["chunk_id"],
                    "raw_archive": None,
                    "raw_member": None,
                    "mask_archive": None,
                    "mask_member": None,
                    "keypoint_csv_archive": None,
                    "keypoint_csv_member": None,
                },
            )
            archive_key = f"{modality}_archive"
            member_key = f"{modality}_member"
            if row[archive_key] is not None:
                errors.append("duplicate_{}:{}".format(modality, row["canonical_id"]))
                continue
            row[archive_key] = str(archive.resolve())
            row[member_key] = member

    aliases = load_aliases(aliases_path)
    pending_keys: set[tuple[str, str, int, str]] = set()
    confirmed: dict[tuple[str, str, int, str], str] = {}
    for alias in aliases:
        activity = (alias.get("activity_label") or "").strip()
        scene = (alias.get("scene_id") or "").strip()
        chunk = int(alias.get("chunk_id") or 0)
        raw_id = (alias.get("raw_id") or "").strip()
        paired_id = (alias.get("paired_id") or "").strip()
        verification = (alias.get("verification") or "").strip().lower()
        raw_key = (activity, scene, chunk, raw_id)
        paired_key = (activity, scene, chunk, paired_id)
        if verification == "manual_confirmed":
            confirmed[raw_key] = paired_id
        elif verification:
            pending_keys.update([raw_key, paired_key])

    items = []
    for row in records.values():
        raw = row["raw_member"] is not None
        mask = row["mask_member"] is not None
        csv_exists = row["keypoint_csv_member"] is not None
        status = determine_pair_status(raw, mask, csv_exists)
        row_key = (
            row["activity_label"],
            row["scene_id"],
            row["chunk_id"],
            row["recording_id"],
        )
        if row_key in pending_keys:
            status = "filename_mismatch_candidate"
        if row_key in confirmed:
            status = "manual_alias_confirmed"
        row.update(
            {
                "pair_status": status,
                "alias_target": confirmed.get(row_key),
                "raw_video_exists": raw,
                "mask_video_exists": mask,
                "provided_keypoint_csv_exists": csv_exists,
                "video_classification_eligible": raw,
                "temporal_annotation_complete": False,
                "split_group_resolved": False,
                "subject_disjoint_split_ready": False,
                "observed_pose_samples_available": False,
                "pre_context_sufficient": False,
                "pre_onset_observed_samples": None,
                "pre_onset_ready": False,
                "event_evaluable": False,
                "temporal_tcn_eligible": False,
                "provisional_split_group": "fallvision_archive:"
                + ":".join(
                    [row["activity_label"], row["scene_id"], str(row["chunk_id"])]
                ),
            }
        )
        items.append(row)

    items.sort(
        key=lambda row: (
            row["activity_label"],
            row["scene_id"],
            row["chunk_id"],
            row["recording_id"],
        )
    )
    return {
        "schema_version": "fallvision_canonical_inventory_v1",
        "source_root": str(files_root.resolve()),
        "alias_table": str(aliases_path.resolve()),
        "archive_count": archive_count,
        "record_count": len(items),
        "raw_video_count": sum(row["raw_video_exists"] for row in items),
        "mask_video_count": sum(row["mask_video_exists"] for row in items),
        "provided_keypoint_csv_count": sum(
            row["provided_keypoint_csv_exists"] for row in items
        ),
        "pair_status_counts": dict(
            sorted(Counter(row["pair_status"] for row in items).items())
        ),
        "temporal_tcn_eligible_count": sum(
            row["temporal_tcn_eligible"] for row in items
        ),
        "errors": errors,
        "items": items,
    }


def write_flat_csv(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(items[0]) if items else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files-root",
        type=Path,
        default=project / "external_datasets" / "fallvision" / "files",
    )
    parser.add_argument(
        "--unrar",
        type=Path,
        default=Path(
            "/home/dmc/.local/dmc_pose_tools/unrar/usr/bin/unrar-nonfree"
        ),
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        default=project
        / "external_datasets"
        / "manifests"
        / "fallvision_pair_aliases.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=project
        / "external_datasets"
        / "manifests"
        / "fallvision_canonical_inventory_v1.json",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=project
        / "external_datasets"
        / "manifests"
        / "fallvision_canonical_inventory_v1.csv",
    )
    args = parser.parse_args()

    payload = build_inventory(args.files_root, args.unrar, args.aliases)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_flat_csv(args.out_csv, payload["items"])
    print(json.dumps({k: v for k, v in payload.items() if k != "items"}, indent=2))
    return 2 if payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
