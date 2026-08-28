#!/usr/bin/env python3
"""Atomically deploy the reviewed GRU as telemetry-only company-core shadow."""
from __future__ import annotations

import argparse
import grp
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

SOURCE = Path("/home/dmc/AI/DMC_POSE_source")
RUNTIME = Path("/opt/.company-core/runtime")
UNIT_DROPIN = Path("/etc/systemd/system/company-core.service.d/20-central-gru-shadow.conf")
BACKUP_ROOT = Path("/var/lib/.company-core/backups")
HOST_POSE_PYTHON = Path("/home/dmc/anaconda3/envs/pose-cuda/bin/python")
MODEL_REL = Path("runs/temporal_gru/usb_reviewed_staged_shadow_v2_20260824/model.pt")
REPORT_REL = Path("runs/temporal_gru/usb_reviewed_staged_shadow_v2_20260824/report.json")
ONNX_REL = Path("runs/temporal_gru/usb_reviewed_staged_shadow_v2_20260824/model_shadow.onnx")
EXTENSION_DIR = SOURCE / "build/protected_shadow_gru_v2"
EXTENSION_FILES = (
    "server_all_cameras.cpython-311-x86_64-linux-gnu.so",
    "live_temporal.cpython-311-x86_64-linux-gnu.so",
    "temporal_model.cpython-311-x86_64-linux-gnu.so",
    "temporal_features.cpython-311-x86_64-linux-gnu.so",
    "temporal_sequence.cpython-311-x86_64-linux-gnu.so",
)
PROTECTED_MANIFEST = RUNTIME / ".protected_core.sha256"
EXTENSION_PINNED = {
    "server_all_cameras.cpython-311-x86_64-linux-gnu.so": "bd30b1f1895fb38eb8b00b8440e96d913721b3e040345accab9be77cff4bfb9b",
    "live_temporal.cpython-311-x86_64-linux-gnu.so": "07fb4d116fd2b84d0f74168321bf916a3d844cda6d8101d1d8504c896a25d3aa",
    "temporal_model.cpython-311-x86_64-linux-gnu.so": "01f85f707df1e40f5115c6b2546a7ead5f196ae0761d502191493e90116f554b",
    "temporal_features.cpython-311-x86_64-linux-gnu.so": "17a73e50844cb3ce80fa0d11b2c2263b37c47cf1a7d15629c45f0f47d7bf3ad4",
    "temporal_sequence.cpython-311-x86_64-linux-gnu.so": "17c03273fd35f46340ea9eeae6b351866ca925f351920babdf2281828d7ec42f",
}
PINNED = {
    str(MODEL_REL): "55673cc2a2770187554855b65d14ca4d2a522f2e20f735e7166f23d7412070ea",
    str(REPORT_REL): "8832ba07e1f6b8288dde9690006c77fa7a8f8c3e43e345cabf5a6bac0afef630",
    str(ONNX_REL): "689d34d0c49e3478b964c6d06f376656991fa1afa67be7ed81ec27c079b1b149",
}
DROPIN = """[Service]
Environment=POSE_TCN_SHADOW=1
Environment=POSE_TCN_MODEL=/opt/.company-core/runtime/runs/temporal_gru/usb_reviewed_staged_shadow_v2_20260824/model.pt
Environment=POSE_TCN_REPORT=/opt/.company-core/runtime/runs/temporal_gru/usb_reviewed_staged_shadow_v2_20260824/report.json
Environment=POSE_TCN_DEVICE=cuda
Environment=POSE_TCN_ALLOW_NON_PROMOTION=1
Environment=POSE_TCN_FUSION_ENABLED=0
Environment=POSE_CENTRAL_ALWAYS_ON=1
Environment=POSE_MOTION_WATCHER_ENABLED=0
Environment=POSE_OCCUPIED_POSE_INTERVAL_SEC=0.001
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    *args: str, check: bool = True, cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, check=check, text=True, capture_output=True, cwd=cwd,
    )


def validate_source() -> dict:
    source_paths = [EXTENSION_DIR / path for path in EXTENSION_FILES]
    source_paths.extend(SOURCE / path for path in (MODEL_REL, REPORT_REL, ONNX_REL))
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")
    for relative, expected in PINNED.items():
        actual = sha256(SOURCE / relative)
        if actual != expected:
            raise ValueError(f"pinned artifact hash mismatch: {relative}: {actual}")
    for filename, expected in EXTENSION_PINNED.items():
        actual = sha256(EXTENSION_DIR / filename)
        if actual != expected:
            raise ValueError(f"pinned extension hash mismatch: {filename}: {actual}")
    report = json.loads((SOURCE / REPORT_REL).read_text(encoding="utf-8"))
    if report.get("model") != "gru_v1":
        raise ValueError("report architecture is not gru_v1")
    if report.get("run_purpose") != "shadow_candidate":
        raise ValueError("report is not a shadow_candidate")
    if report.get("promotion_eligible") is not False:
        raise ValueError("report must be promotion_eligible=false")
    if (report.get("window_rows"), report.get("sample_hz")) != (80, 20.0):
        raise ValueError("report is not 80x109 @ 20Hz")
    return report


def install(
    source: Path, target: Path, *, mode: int | None = None,
    owner: tuple[int, int] | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".new")
    shutil.copy2(source, temporary)
    if mode is not None:
        temporary.chmod(mode)
    if owner is not None:
        os.chown(temporary, *owner)
    os.replace(temporary, target)


def update_protected_manifest() -> None:
    lines = PROTECTED_MANIFEST.read_text(encoding="utf-8").splitlines()
    entries: list[tuple[str, str]] = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid protected manifest line: {line!r}")
        entries.append((parts[0], parts[1].lstrip("*")))
    replacements = {
        filename: sha256(RUNTIME / filename) for filename in EXTENSION_FILES
    }
    replacements.update({
        str(relative): sha256(RUNTIME / relative)
        for relative in (MODEL_REL, REPORT_REL, ONNX_REL)
    })
    seen = set()
    updated = []
    for digest, relative in entries:
        if relative in replacements:
            digest = replacements[relative]
            seen.add(relative)
        updated.append((digest, relative))
    for relative in (str(MODEL_REL), str(REPORT_REL), str(ONNX_REL)):
        if relative not in seen and all(path != relative for _, path in updated):
            updated.append((replacements[relative], relative))
    missing_extensions = set(EXTENSION_FILES) - {
        relative for _, relative in updated
    }
    if missing_extensions:
        raise ValueError(
            f"protected manifest did not contain extensions: {sorted(missing_extensions)}"
        )
    temporary = PROTECTED_MANIFEST.with_name(PROTECTED_MANIFEST.name + ".new")
    temporary.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in updated),
        encoding="utf-8",
    )
    temporary.chmod(0o640)
    os.chown(temporary, 0, grp.getgrnam("company-core").gr_gid)
    os.replace(temporary, PROTECTED_MANIFEST)
    run("sha256sum", "--check", PROTECTED_MANIFEST.name, cwd=RUNTIME)


def wait_ready(timeout_sec: int = 150) -> dict:
    deadline = time.monotonic() + timeout_sec
    last_error = "not started"
    http = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        active = run("systemctl", "is-active", "company-core.service", check=False)
        if active.stdout.strip() == "active":
            try:
                with http.open(
                    "http://127.0.0.1:8030/api/cameras", timeout=8,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    rows = payload if isinstance(payload, list) else payload.get("cameras", [])
                    if len(rows) != 6 or not all(row.get("online") for row in rows):
                        last_error = f"camera readiness={len(rows)}/6"
                    else:
                        log_path = Path("/var/log/.company-core/core.log")
                        log_tail = log_path.read_text(
                            encoding="utf-8", errors="replace"
                        )[-200000:]
                        if str(RUNTIME / MODEL_REL) not in log_tail:
                            last_error = "new GRU load message not found in core.log"
                        else:
                            return {
                                "camera_count": len(rows),
                                "online_count": sum(bool(row.get("online")) for row in rows),
                                "payload_type": type(payload).__name__,
                                "model_load_log_verified": True,
                            }
            except Exception as exc:
                last_error = f"camera API: {exc}"
        else:
            last_error = f"service={active.stdout.strip()}"
        time.sleep(2)
    raise TimeoutError(f"company-core did not become ready: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    report = validate_source()
    plan = {
        "source": str(SOURCE), "runtime": str(RUNTIME),
        "runtime_files": list(EXTENSION_FILES), "artifact_files": list(PINNED),
        "authority": "telemetry_only", "fusion_enabled": False,
        "always_on_pose": True, "sample_hz": report["sample_hz"],
        "window_rows": report["window_rows"],
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    if os.geteuid() != 0:
        raise PermissionError("run with sudo")
    for relative in EXTENSION_FILES:
        target = RUNTIME / relative
        if not target.is_file():
            raise FileNotFoundError(f"refusing deploy; runtime file missing: {target}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = BACKUP_ROOT / f"central_gru_shadow_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    deployed = []
    had_dropin = UNIT_DROPIN.exists()
    artifact_existed = {
        relative: (RUNTIME / relative).exists()
        for relative in (MODEL_REL, REPORT_REL, ONNX_REL)
    }
    manifest_backup = backup / ".protected_core.sha256"
    shutil.copy2(PROTECTED_MANIFEST, manifest_backup)
    core_group = grp.getgrnam("company-core").gr_gid
    try:
        for relative in EXTENSION_FILES:
            source, target = EXTENSION_DIR / relative, RUNTIME / relative
            backup_target = backup / relative
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_target)
            install(source, target, mode=0o640, owner=(0, core_group))
            deployed.append({"path": str(target), "sha256": sha256(target)})
        for relative in (MODEL_REL, REPORT_REL, ONNX_REL):
            target = RUNTIME / relative
            if target.exists():
                backup_target = backup / relative
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup_target)
            install(SOURCE / relative, target, mode=0o640, owner=(0, core_group))
            deployed.append({"path": str(target), "sha256": sha256(target)})
        update_protected_manifest()
        deployed.append({
            "path": str(PROTECTED_MANIFEST),
            "sha256": sha256(PROTECTED_MANIFEST),
        })
        run(
            str(HOST_POSE_PYTHON), "-c",
            "import temporal_sequence,temporal_features,temporal_model,live_temporal,server_all_cameras",
            cwd=RUNTIME,
        )
        if had_dropin:
            shutil.copy2(UNIT_DROPIN, backup / "20-central-gru-shadow.conf")
        UNIT_DROPIN.parent.mkdir(parents=True, exist_ok=True)
        temporary = UNIT_DROPIN.with_suffix(".conf.new")
        temporary.write_text(DROPIN, encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, UNIT_DROPIN)
        run("systemctl", "daemon-reload")
        run("systemctl", "restart", "company-core.service")
        health = wait_ready()
    except Exception:
        for relative in EXTENSION_FILES:
            saved = backup / relative
            if saved.exists():
                install(
                    saved, RUNTIME / relative,
                    mode=0o640, owner=(0, core_group),
                )
        if manifest_backup.exists():
            install(
                manifest_backup, PROTECTED_MANIFEST,
                mode=0o640, owner=(0, core_group),
            )
        for relative, existed in artifact_existed.items():
            saved = backup / relative
            target = RUNTIME / relative
            if saved.exists():
                install(
                    saved, target, mode=0o640, owner=(0, core_group),
                )
            elif not existed and target.exists():
                target.unlink()
        if had_dropin:
            install(
                backup / "20-central-gru-shadow.conf", UNIT_DROPIN,
                mode=0o644, owner=(0, 0),
            )
        elif UNIT_DROPIN.exists():
            UNIT_DROPIN.unlink()
        run("systemctl", "daemon-reload", check=False)
        run("systemctl", "restart", "company-core.service", check=False)
        raise

    record = {
        "schema_version": "dmc_central_gru_shadow_deployment_v1",
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "authority": "telemetry_only", "fusion_enabled": False,
        "promotion_eligible": False, "backup": str(backup),
        "dropin": str(UNIT_DROPIN), "health": health, "files": deployed,
    }
    (backup / "deployment.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
