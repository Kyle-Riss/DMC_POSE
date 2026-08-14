"""Non-blocking, single-flight replay execution."""

from dataclasses import dataclass
from threading import Lock, Thread
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ReplayCompletion(Generic[T]):
    value: T | None = None
    error: BaseException | None = None


class AsyncReplayWorker(Generic[T]):
    """Run at most one daemon replay job and expose one-shot completion."""

    def __init__(self, name: str):
        self._name = name
        self._lock = Lock()
        self._running = False
        self._completion: ReplayCompletion[T] | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def submit(self, job: Callable[[], T]) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._completion = None

        def execute() -> None:
            try:
                completion = ReplayCompletion(value=job())
            except BaseException as exc:
                completion = ReplayCompletion[T](error=exc)
            with self._lock:
                self._completion = completion
                self._running = False

        Thread(
            target=execute,
            name=f"replay-{self._name}",
            daemon=True,
        ).start()
        return True

    def poll(self) -> ReplayCompletion[T] | None:
        with self._lock:
            completion = self._completion
            self._completion = None
            return completion

