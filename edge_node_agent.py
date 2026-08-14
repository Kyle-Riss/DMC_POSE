"""Minimal Pi agent for durable heartbeat/result delivery to the central API."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread

from edge_contract_v1 import EdgeCapabilities, EdgeHeartbeat, EdgeInferenceResult, utc_now
from edge_motion_watcher import EdgeMotionWatcher
from edge_outbox_v1 import AsyncOutboxWriter, EdgeOutbox, EdgeOutboxSender
from edge_pose_shadow import EdgePoseShadow


def boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.exists() else f"process-{os.getpid()}"


def uptime_sec() -> float:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except Exception:
        return 0.0


def storage_free_mb(path: str | Path) -> float:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize / (1024 * 1024)


def tcp_port_open(host: str, port: int, timeout_sec: float = 0.25) -> bool:
    """Return whether a local capture endpoint accepts TCP connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def process_names() -> set[str]:
    """Read process names without exposing command lines or RTSP credentials."""
    names: set[str] = set()
    for comm in Path("/proc").glob("[0-9]*/comm"):
        try:
            names.add(comm.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeError):
            continue
    return names


class EdgeNodeAgent:
    """Control-plane agent; camera/model runtime updates its public state."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.node_id = str(config["node_id"])
        self.camera_id = str(config["camera_id"])
        self.server_url = str(config["server_url"]).rstrip("/")
        token_file = config.get("api_token_file")
        self.api_token = str(config.get("api_token", "")).strip()
        if token_file:
            self.api_token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
        if len(self.api_token) < 32:
            raise ValueError("api_token or api_token_file with at least 32 characters is required")
        self.software_version = str(config.get("software_version", "edge-agent-v1"))
        self.model_bundle_version = config.get("model_bundle_version")
        self.boot_id = boot_id()
        self.state = {
            "capture_connected": False,
            "capture_fps": 0.0,
            "watcher_fps": 0.0,
            "runtime_mode": "DEGRADED",
            "roi_state": "UNAVAILABLE",
            "roi_version": 0,
        }
        self.capabilities = EdgeCapabilities.model_validate(config.get("capabilities", {}))
        spool = Path(config.get("spool_path", "runtime_data/edge_node/outbox.sqlite3"))
        self.sequence_path = Path(
            config.get("sequence_path", spool.parent / "heartbeat_sequence")
        )
        try:
            self.sequence = max(0, int(self.sequence_path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            self.sequence = 0
        self.pose_result_sequence_path = Path(config.get(
            "pose_result_sequence_path", spool.parent / "pose_result_sequence"
        ))
        try:
            self.pose_result_sequence = max(
                0, int(self.pose_result_sequence_path.read_text(encoding="utf-8").strip())
            )
        except (OSError, ValueError):
            self.pose_result_sequence = 0
        self.pose_shadow_model_bundle_version = str(
            config.get("pose_shadow_model_bundle_version", "cm4-onnx-candidate-v1")
        )
        self.outbox = EdgeOutbox(spool)
        self.writer = AsyncOutboxWriter(self.outbox)
        self.sender = EdgeOutboxSender(self.outbox)
        self.sender_poll_sec = max(0.05, float(config.get("sender_poll_sec", 0.1)))
        self._sender_stop = Event()
        self._sender_wakeup = Event()
        self._sender_thread: Thread | None = None
        watcher_config = config.get("motion_watcher_config")
        self.motion_watcher = (
            EdgeMotionWatcher(**watcher_config) if watcher_config else None
        )
        pose_shadow_config = config.get("pose_shadow_config")
        self.pose_shadow = (
            EdgePoseShadow(self.motion_watcher, **pose_shadow_config)
            if pose_shadow_config and self.motion_watcher is not None
            else None
        )
        if self.pose_shadow is not None:
            self.pose_shadow.on_result = self.queue_pose_shadow_result

    def queue_pose_shadow_result(self, event: dict) -> EdgeInferenceResult:
        person_present = int(event.get("detection_count", 0)) > 0
        score = float(event.get("best_person_score", 0.0))
        visible = int(event.get("best_visible_keypoints", 0))
        payload = EdgeInferenceResult(
            node_id=self.node_id, camera_id=self.camera_id, boot_id=self.boot_id,
            frame_seq=self.pose_result_sequence,
            captured_at=datetime.fromtimestamp(
                float(event.get("captured_unix", event["event_unix"])), tz=timezone.utc
            ),
            model_bundle_version=self.pose_shadow_model_bundle_version,
            roi_version=0, primary_track_id=None,
            person_present=person_present, body_in_bed_ratio=0.0,
            pose_label="person_detected" if person_present else None,
            pose_confidence=score, temporal_ready=False, temporal_samples=0,
            temporal_probability=0.0, temporal_candidate=False,
            fusion_phase="INSUFFICIENT", fusion_risk=0.0,
            evidence=["motion_burst", "pose_shadow", "pose_model:yolo11n"],
            quality=min(1.0, visible / 17.0) * score,
            inference_latency_ms=float(event.get("snapshot_and_pose_ms", 0.0)),
        )
        self.pose_result_sequence += 1
        self.pose_result_sequence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.pose_result_sequence_path.with_suffix(".tmp")
        temporary.write_text(str(self.pose_result_sequence), encoding="utf-8")
        os.replace(temporary, self.pose_result_sequence_path)
        if not self.writer.submit(payload):
            raise RuntimeError("pose shadow result queue is full")
        self._sender_wakeup.set()
        return payload

    def refresh_capture_health(self) -> None:
        health = self.config.get("capture_health")
        if not health:
            return
        host = str(health.get("tcp_host", "127.0.0.1"))
        port = int(health.get("tcp_port", 8554))
        required = {str(name) for name in health.get("required_processes", [])}
        processes_ok = required.issubset(process_names())
        self.state["capture_connected"] = tcp_port_open(host, port) and processes_ok

    def refresh_motion_state(self) -> None:
        if self.motion_watcher is None:
            return
        motion = self.motion_watcher.status()
        self.state["watcher_fps"] = float(motion["watcher_fps"])
        self.state["runtime_mode"] = "BURST" if motion["burst_active"] else "EMPTY"

    def update_runtime(self, **state) -> None:
        allowed = set(self.state)
        unknown = set(state) - allowed
        if unknown:
            raise ValueError(f"unknown runtime state: {sorted(unknown)}")
        self.state.update(state)

    def heartbeat(self) -> EdgeHeartbeat:
        self.refresh_capture_health()
        self.refresh_motion_state()
        spool = self.outbox.stats()
        payload = EdgeHeartbeat(
            node_id=self.node_id,
            camera_id=self.camera_id,
            boot_id=self.boot_id,
            sequence=self.sequence,
            sent_at=utc_now(),
            software_version=self.software_version,
            model_bundle_version=self.model_bundle_version,
            uptime_sec=uptime_sec(),
            capture_connected=bool(self.state["capture_connected"]),
            capture_fps=float(self.state["capture_fps"]),
            watcher_fps=float(self.state["watcher_fps"]),
            runtime_mode=self.state["runtime_mode"],
            roi_state=self.state["roi_state"],
            roi_version=int(self.state["roi_version"]),
            spool_depth=spool["pending"],
            spool_bytes=spool["payload_bytes"],
            storage_free_mb=storage_free_mb(self.outbox.path.parent),
            capabilities=self.capabilities,
        )
        self.sequence += 1
        self.sequence_path.parent.mkdir(parents=True, exist_ok=True)
        sequence_tmp = self.sequence_path.with_suffix(".tmp")
        sequence_tmp.write_text(str(self.sequence), encoding="utf-8")
        os.replace(sequence_tmp, self.sequence_path)
        return payload

    def _post(self, endpoint: str, payload: dict) -> bool:
        request = urllib.request.Request(
            self.server_url + endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, TimeoutError):
            return False

    def cycle(self, *, flush: bool = True) -> dict:
        queued = self.writer.submit(self.heartbeat())
        # The writer owns SQLite; give it a scheduling opportunity before flush.
        self.writer.queue.join()
        self._sender_wakeup.set()
        result = (
            self.sender.flush_once(self._post, limit=64)
            if flush else {"sent": 0, "failed": 0, **self.outbox.stats()}
        )
        result["heartbeat_queued"] = queued
        return result

    def start_sender(self) -> None:
        if self._sender_thread and self._sender_thread.is_alive():
            return
        self._sender_stop.clear()
        self._sender_thread = Thread(
            target=self._sender_loop, name="edge-outbox-sender", daemon=True
        )
        self._sender_thread.start()

    def stop_sender(self, timeout: float = 3.0) -> None:
        self._sender_stop.set()
        self._sender_wakeup.set()
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=max(0.0, float(timeout)))

    def _sender_loop(self) -> None:
        while not self._sender_stop.is_set():
            self.writer.queue.join()
            self.sender.flush_once(self._post, limit=64)
            self._sender_wakeup.wait(self.sender_poll_sec)
            self._sender_wakeup.clear()
        self.writer.queue.join()
        self.sender.flush_once(self._post, limit=64)

    def run(self) -> None:
        interval = max(1.0, float(self.config.get("heartbeat_interval_sec", 5.0)))
        running = True

        def stop(*_):
            nonlocal running
            running = False

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        self.writer.start()
        self.start_sender()
        if self.motion_watcher is not None:
            self.motion_watcher.start()
        if self.pose_shadow is not None:
            self.pose_shadow.start()
        try:
            while running:
                self.cycle(flush=False)
                time.sleep(interval)
        finally:
            if self.pose_shadow is not None:
                self.pose_shadow.stop()
            if self.motion_watcher is not None:
                self.motion_watcher.stop()
            self.stop_sender()
            self.writer.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    agent = EdgeNodeAgent(config)
    agent.writer.start()
    try:
        if args.once:
            print(json.dumps(agent.cycle(), indent=2))
        else:
            agent.run()
    finally:
        agent.writer.stop()


if __name__ == "__main__":
    main()
