#!/usr/bin/env python3
"""Extract FallVision-provided keypoint CSV archives once, with an audit report."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def archives_from_inventory(path: Path) -> list[Path]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = {
            row["keypoint_csv_archive"]
            for row in csv.DictReader(handle)
            if row.get("provided_keypoint_csv_exists") == "True"
            and row.get("keypoint_csv_archive")
        }
    return sorted(Path(value) for value in values)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=project / "external_datasets/manifests/fallvision_canonical_inventory_v1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "external_datasets/fallvision/provided_keypoints",
    )
    parser.add_argument(
        "--unrar",
        type=Path,
        default=Path("/home/dmc/.local/dmc_pose_tools/unrar/usr/bin/unrar-nonfree"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project / "external_datasets/fallvision/provided_keypoints_extraction_report.json",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archives = archives_from_inventory(args.inventory)
    results = []
    for index, archive in enumerate(archives, 1):
        command = [str(args.unrar), "x", "-o-", "-inul", str(archive), str(args.output_dir) + "/"]
        completed = subprocess.run(command, capture_output=True, text=True)
        result = {
            "archive": str(archive),
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
        results.append(result)
        print(f"[{index:02d}/{len(archives):02d}] rc={completed.returncode} {archive.name}", flush=True)
    csv_count = sum(1 for _ in args.output_dir.rglob("*.csv"))
    report = {
        "schema_version": "fallvision_provided_keypoints_extraction_v1",
        "inventory": str(args.inventory.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "archive_count": len(archives),
        "successful_archive_count": sum(item["returncode"] == 0 for item in results),
        "csv_count": csv_count,
        "results": results,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if any(item["returncode"] != 0 for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
