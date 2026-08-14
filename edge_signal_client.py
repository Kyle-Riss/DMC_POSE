"""Authenticated, read-only bridge from edge control results to central scheduling."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event, Lock, Thread
import time
import urllib.error
import urllib.request


class EdgeSignalClient:
    """Poll latest edge facts without trusting edge temporal/fusion decisions."""

    def __init__(
        self,
        server_url: str,
        token_file: str | Path,
        *,
        poll_interval_sec: float = 0.25,
        max_result_age_sec: float = 4.0,
        min_person_confidence: float = 0.25,
        min_quality: float = 0.20,
    ):
        self.server_url = str(server_url).rstrip("/")
        self.token_file = Path(token_file)
        self.poll_interval_sec = max(0.05, float(poll_interval_sec))
        self.max_result_age_sec = max(0.1, float(max_result_age_sec))
        self.min_person_confidence = float(min_person_confidence)
        self.min_quality = float(min_quality)
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._results: dict[str, dict] = {}
        self._last_poll_unix: float | None = None
        self._last_success_unix: float | None = None
        self._error_total = 0
        self._last_error = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="edge-signal-client", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @staticmethod
    def _unix(timestamp: str) -> float:
        value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()

    def poll_once(self) -> int:
        token = self.token_file.read_text(encoding="utf-8").strip()
        request = urllib.request.Request(
            self.server_url + "/edge/nodes",
            headers={"Authorization": f"Bearer {token}"},
        )
        now = time.time()
        with urllib.request.urlopen(request, timeout=1.0) as response:
            payload = json.load(response)
        latest = {}
        for node in payload.get("nodes", []):
            result = node.get("latest_result")
            camera_id = node.get("camera_id")
            if camera_id and result:
                item = dict(result)
                item["captured_unix"] = self._unix(str(item["captured_at"]))
                latest[str(camera_id)] = item
        with self._lock:
            self._results = latest
            self._last_poll_unix = now
            self._last_success_unix = time.time()
            self._last_error = ""
        return len(latest)

    def status(self, camera_id: str, *, now_unix: float | None = None) -> dict:
        now = time.time() if now_unix is None else float(now_unix)
        with self._lock:
            result = dict(self._results.get(str(camera_id), {}))
            last_success = self._last_success_unix
            error_total = self._error_total
            last_error = self._last_error
        captured = result.get("captured_unix")
        age_sec = None if captured is None else max(0.0, now - float(captured))
        fresh = age_sec is not None and age_sec <= self.max_result_age_sec
        person = bool(result.get("person_present", False))
        confidence = float(result.get("pose_confidence", 0.0))
        quality = float(result.get("quality", 0.0))
        wake = (
            fresh and person
            and confidence >= self.min_person_confidence
            and quality >= self.min_quality
        )
        return {
            "thread_alive": self.is_alive(),
            "connected": last_success is not None and now - last_success <= 2.0,
            "wake_active": wake,
            "result_fresh": fresh,
            "result_age_ms": None if age_sec is None else age_sec * 1000.0,
            "person_present": person if fresh else False,
            "last_person_present": person,
            "pose_confidence": confidence,
            "quality": quality,
            "frame_seq": result.get("frame_seq"),
            "model_bundle_version": result.get("model_bundle_version"),
            "error_total": error_total,
            "last_error": last_error,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except (OSError, ValueError, KeyError, json.JSONDecodeError,
                    urllib.error.URLError) as exc:
                with self._lock:
                    self._last_poll_unix = time.time()
                    self._error_total += 1
                    self._last_error = str(exc)[:300]
            self._stop.wait(self.poll_interval_sec)
