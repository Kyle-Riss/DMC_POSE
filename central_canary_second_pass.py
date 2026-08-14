"""Event-triggered verifier backed by the central reference pipeline during canary.

It never opens RTSP.  It snapshots the already-running reference pipeline only
after an edge event, and inventories bounded evidence frames uploaded by the Pi.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

from edge_contract_v1 import EdgeEventStart


def load_reference_status(url: str) -> dict:
    with urlopen(url, timeout=3.0) as response:
        return json.load(response)


class CentralCanarySecondPass:
    def __init__(
        self,
        frame_root: str | Path,
        *,
        status_url: str = "http://127.0.0.1:8000/status",
        frame_wait_sec: float = 2.0,
        status_loader: Callable[[str], dict] = load_reference_status,
    ):
        self.frame_root = Path(frame_root)
        self.status_url = status_url
        self.frame_wait_sec = max(0.0, float(frame_wait_sec))
        self.status_loader = status_loader

    def _frames(self, event_id: str) -> list[Path]:
        root = (self.frame_root / event_id).resolve()
        if root.parent != self.frame_root.resolve() or not root.is_dir():
            return []
        return sorted(root.glob("*.jpg"))[:180]

    def __call__(self, event: EdgeEventStart) -> dict:
        deadline = time.monotonic() + self.frame_wait_sec
        frames = self._frames(event.event_id)
        while not frames and time.monotonic() < deadline:
            time.sleep(0.1)
            frames = self._frames(event.event_id)

        reference = None
        reference_error = None
        try:
            reference = self.status_loader(self.status_url).get(event.camera_id)
        except Exception as exc:
            reference_error = type(exc).__name__

        risk = float((reference or {}).get("fusion_risk", 0.0))
        phase = (reference or {}).get("fusion_phase")
        if phase in {"VERIFY", "SHADOW_ALERT"} or risk >= 0.7:
            decision = "central_reference_confirmed"
        elif reference is not None:
            decision = "central_reference_not_confirmed"
        elif frames:
            decision = "evidence_received_analyzer_pending"
        else:
            decision = "insufficient_evidence"

        frame_hashes = []
        for path in frames[:5]:
            frame_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        keep = {
            key: (reference or {}).get(key)
            for key in (
                "fusion_phase", "fusion_risk", "fusion_quality", "fusion_evidence",
                "pose_class", "pose_confidence", "tcn_ready", "tcn_probability",
                "runtime_mode", "primary_track_id", "bed_roi_version",
            )
            if key in (reference or {})
        }
        return {
            "decision": decision,
            "mode": "central_reference_snapshot_canary",
            "continuous_stream_opened": False,
            "evidence_frame_count": len(frames),
            "evidence_frame_sha256_sample": frame_hashes,
            "reference": keep,
            "reference_error": reference_error,
        }
