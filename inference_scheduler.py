"""Latest-only central scheduler for GPU inference requests.

Camera analysis loops never queue an unbounded number of frames.  Each
``(model, camera)`` mailbox owns at most one waiting request; a newer request
supersedes an older one.  A single worker selects work by priority while a
small urgent quota prevents low-priority probes and bed refreshes from
starving forever.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable


P0_VERIFY = 0
P1_BURST = 1
P2_OCCUPIED = 2
P3_EMPTY_PROBE = 3
P4_BED_SEG = 4


@dataclass
class InferenceOutcome:
    result: Any = None
    completed: bool = False
    dropped: bool = False
    drop_reason: str | None = None
    queue_latency_ms: float = 0.0
    inference_ms: float = 0.0


@dataclass
class _Request:
    model: str
    camera_id: str
    frame: Any
    frame_seq: int
    priority: int
    created_mono: float
    deadline_mono: float
    done: threading.Event = field(default_factory=threading.Event)
    outcome: InferenceOutcome = field(default_factory=InferenceOutcome)
    cancelled: bool = False


class LatestInferenceScheduler:
    """Serialize model access while keeping only useful, recent work."""

    def __init__(
        self,
        infer_pose: Callable[[Any], Any],
        infer_seg: Callable[[Any], Any],
        *,
        infer_pose_replay: Callable[[Any], Any] | None = None,
        urgent_quota: int = 4,
        metrics_window_sec: float = 30.0,
    ):
        self._infer = {
            "pose": infer_pose,
            "pose_replay": infer_pose_replay or infer_pose,
            "seg": infer_seg,
        }
        self._urgent_quota = max(1, int(urgent_quota))
        self._metrics_window_sec = max(1.0, float(metrics_window_sec))
        self._condition = threading.Condition()
        self._mailboxes: dict[tuple[str, str], _Request] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._urgent_streak = 0
        self._started_mono = time.monotonic()
        self._stats = defaultdict(self._new_stats)

    @staticmethod
    def _new_stats():
        return {
            "submitted_total": 0,
            "completed_total": 0,
            "stale_drop_total": 0,
            "superseded_drop_total": 0,
            "timeout_total": 0,
            "error_total": 0,
            "last_queue_latency_ms": 0.0,
            "last_inference_ms": 0.0,
            "last_priority": None,
            "last_model": None,
            "completed_times": deque(),
        }

    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                name="central-inference-scheduler",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._running = False
            pending = list(self._mailboxes.values())
            self._mailboxes.clear()
            for request in pending:
                self._drop_locked(request, "shutdown")
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def request(
        self,
        model: str,
        camera_id: str,
        frame: Any,
        *,
        frame_seq: int,
        priority: int,
        deadline_sec: float,
    ) -> InferenceOutcome:
        if model not in self._infer:
            raise ValueError(f"unknown inference model: {model}")
        now = time.monotonic()
        request = _Request(
            model=model,
            camera_id=camera_id,
            frame=frame,
            frame_seq=int(frame_seq),
            priority=int(priority),
            created_mono=now,
            deadline_mono=now + max(0.01, float(deadline_sec)),
        )
        key = (model, camera_id)
        with self._condition:
            if not self._running:
                return InferenceOutcome(dropped=True, drop_reason="not_running")
            previous = self._mailboxes.get(key)
            if previous is not None:
                self._stats[camera_id]["superseded_drop_total"] += 1
                self._drop_locked(previous, "superseded")
            self._mailboxes[key] = request
            self._stats[camera_id]["submitted_total"] += 1
            self._condition.notify()

        wait_sec = max(0.02, request.deadline_mono - time.monotonic() + 0.25)
        if not request.done.wait(wait_sec):
            with self._condition:
                request.cancelled = True
                if self._mailboxes.get(key) is request:
                    self._mailboxes.pop(key, None)
                self._stats[camera_id]["timeout_total"] += 1
            return InferenceOutcome(dropped=True, drop_reason="caller_timeout")
        return request.outcome

    def request_pose(self, camera_id: str, frame: Any, **kwargs) -> InferenceOutcome:
        return self.request("pose", camera_id, frame, **kwargs)

    def request_pose_replay(
        self, camera_id: str, frame: Any, **kwargs
    ) -> InferenceOutcome:
        return self.request("pose_replay", camera_id, frame, **kwargs)

    def request_seg(self, camera_id: str, frame: Any, **kwargs) -> InferenceOutcome:
        return self.request("seg", camera_id, frame, **kwargs)

    def _drop_locked(self, request: _Request, reason: str) -> None:
        request.cancelled = True
        request.outcome = InferenceOutcome(dropped=True, drop_reason=reason)
        request.done.set()

    def _remove_expired_locked(self, now: float) -> None:
        for key, request in list(self._mailboxes.items()):
            if request.cancelled:
                self._mailboxes.pop(key, None)
            elif request.deadline_mono <= now:
                self._mailboxes.pop(key, None)
                self._stats[request.camera_id]["stale_drop_total"] += 1
                self._drop_locked(request, "stale")

    def _choose_locked(self) -> _Request | None:
        now = time.monotonic()
        self._remove_expired_locked(now)
        candidates = [r for r in self._mailboxes.values() if not r.cancelled]
        if not candidates:
            return None

        urgent = [r for r in candidates if r.priority <= P1_BURST]
        normal = [r for r in candidates if r.priority > P1_BURST]
        if urgent and (self._urgent_streak < self._urgent_quota or not normal):
            chosen = min(urgent, key=lambda r: (r.priority, r.created_mono))
            self._urgent_streak += 1
        elif normal:
            # After the urgent quota, oldest normal work wins once. This is
            # explicit starvation protection, not FIFO backlog processing.
            chosen = min(normal, key=lambda r: r.created_mono)
            self._urgent_streak = 0
        else:
            chosen = min(candidates, key=lambda r: (r.priority, r.created_mono))
        self._mailboxes.pop((chosen.model, chosen.camera_id), None)
        return chosen

    def _run(self) -> None:
        while True:
            with self._condition:
                request = self._choose_locked()
                while self._running and request is None:
                    self._condition.wait(timeout=0.05)
                    request = self._choose_locked()
                if not self._running and request is None:
                    return
            if request is None or request.cancelled:
                continue

            start = time.monotonic()
            queue_ms = (start - request.created_mono) * 1000.0
            try:
                result = self._infer[request.model](request.frame)
                inference_ms = (time.monotonic() - start) * 1000.0
                outcome = InferenceOutcome(
                    result=result,
                    completed=True,
                    queue_latency_ms=queue_ms,
                    inference_ms=inference_ms,
                )
            except Exception as exc:
                inference_ms = (time.monotonic() - start) * 1000.0
                outcome = InferenceOutcome(
                    dropped=True,
                    drop_reason=f"error:{type(exc).__name__}",
                    queue_latency_ms=queue_ms,
                    inference_ms=inference_ms,
                )
                with self._condition:
                    self._stats[request.camera_id]["error_total"] += 1

            with self._condition:
                stats = self._stats[request.camera_id]
                if outcome.completed:
                    stats["completed_total"] += 1
                    stats["completed_times"].append(time.monotonic())
                stats["last_queue_latency_ms"] = queue_ms
                stats["last_inference_ms"] = inference_ms
                stats["last_priority"] = request.priority
                stats["last_model"] = request.model
                request.outcome = outcome
                request.done.set()

    def metrics(self, camera_id: str) -> dict[str, Any]:
        now = time.monotonic()
        with self._condition:
            stats = self._stats[camera_id]
            times = stats["completed_times"]
            cutoff = now - self._metrics_window_sec
            while times and times[0] < cutoff:
                times.popleft()
            elapsed = max(0.001, now - self._started_mono)
            completed_hz = len(times) / min(self._metrics_window_sec, elapsed)
            pending = sum(
                1 for request in self._mailboxes.values()
                if request.camera_id == camera_id
            )
            return {
                key: value
                for key, value in stats.items()
                if key != "completed_times"
            } | {
                "completed_hz": completed_hz,
                "pending": pending,
                "thread_alive": bool(self._thread and self._thread.is_alive()),
            }
