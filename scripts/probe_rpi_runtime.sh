#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import json
import os
import platform
from pathlib import Path

def read(path, default=None):
    try:
        return Path(path).read_text(errors="replace").strip("\x00\n ")
    except Exception:
        return default

def command_exists(name):
    return any((Path(folder) / name).exists() for folder in os.environ.get("PATH", "").split(":"))

meminfo = {}
for line in (read("/proc/meminfo", "") or "").splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        meminfo[key] = value.strip()

stat = os.statvfs("/")
report = {
    "hostname": platform.node(),
    "machine": platform.machine(),
    "kernel": platform.release(),
    "device_model": read("/proc/device-tree/model", "unknown"),
    "debian_version": read("/etc/debian_version", "unknown"),
    "cpu_count": os.cpu_count(),
    "memory_total": meminfo.get("MemTotal"),
    "root_free_mb": round(stat.f_bavail * stat.f_frsize / 1024 / 1024, 1),
    "boot_id": read("/proc/sys/kernel/random/boot_id"),
    "python": platform.python_version(),
    "commands": {name: command_exists(name) for name in [
        "ffmpeg", "ffprobe", "mediamtx", "rpicam-vid", "libcamera-vid",
        "vcgencmd", "docker", "podman",
    ]},
    "video_devices": sorted(str(path) for path in Path("/dev").glob("video*")),
    "accelerators": sorted(str(path) for pattern in ["hailo*", "dri/*"] for path in Path("/dev").glob(pattern)),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

