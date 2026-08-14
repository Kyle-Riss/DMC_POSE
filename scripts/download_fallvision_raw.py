#!/usr/bin/env python3
"""Download only FallVision Raw Video archives with resume and MD5 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

DOI = "doi:10.7910/DVN/75QPKK"
API = "https://dataverse.harvard.edu/api"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_metadata() -> dict:
    url = f"{API}/datasets/:persistentId/?{urllib.parse.urlencode({'persistentId': DOI})}"
    request = urllib.request.Request(url, headers={"User-Agent": "DMC-POSE-dataset-client/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def raw_files(metadata: dict) -> list[dict]:
    files = metadata["data"]["latestVersion"]["files"]
    return sorted(
        [item for item in files if (item.get("directoryLabel") or "").endswith("Raw Video")],
        key=lambda item: (item.get("directoryLabel") or "", item.get("label") or ""),
    )


def destination(root: Path, item: dict) -> Path:
    parts = (item.get("directoryLabel") or "").split("/")
    # Preserve Fall|No Fall / Bed|Chair|Stand while dropping common prefixes.
    relative = Path(*parts[1:-1]) if len(parts) >= 4 else Path("unknown")
    return root / relative / item["label"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--out-dir", type=Path, default=project_root / "external_datasets/fallvision/raw_archives")
    parser.add_argument("--metadata", type=Path, default=project_root / "external_datasets/fallvision/metadata.json")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--file-id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = fetch_metadata()
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    files = raw_files(metadata)
    if args.file_id is not None:
        files = [item for item in files if int(item["dataFile"]["id"]) == args.file_id]
        if not files:
            raise ValueError(f"raw video file id not found: {args.file_id}")
    if args.max_files is not None:
        files = files[: args.max_files]

    plan = []
    failures = []
    for item in files:
        data_file = item["dataFile"]
        target = destination(args.out_dir, item)
        expected_size = int(data_file.get("filesize") or 0)
        expected_md5 = (data_file.get("checksum") or {}).get("value", "").lower()
        plan.append({"id": data_file["id"], "path": str(target), "bytes": expected_size, "md5": expected_md5})
        if args.dry_run:
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == expected_size and md5(target) == expected_md5:
            print(f"verified: {target}", flush=True)
            continue

        partial = target.with_suffix(target.suffix + ".part")
        url = f"{API}/access/datafile/{data_file['id']}?format=original"
        command = [
            "curl", "-fL", "--retry", "8", "--retry-delay", "5",
            "--continue-at", "-", "--output", str(partial), url,
        ]
        print(f"downloading: {target.name} ({expected_size / 1024**2:.1f} MiB)", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures.append({"path": str(target), "error": f"curl exit {completed.returncode}"})
            continue
        actual_md5 = md5(partial)
        if partial.stat().st_size != expected_size or actual_md5 != expected_md5:
            failures.append({"path": str(target), "error": "size_or_md5_mismatch", "actual_md5": actual_md5})
            continue
        partial.replace(target)
        print(f"verified: {target}", flush=True)

    summary = {
        "selected_files": len(files),
        "selected_bytes": sum(row["bytes"] for row in plan),
        "dry_run": args.dry_run,
        "failures": failures,
        "files": plan,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
