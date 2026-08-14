import threading
import time

from async_replay_worker import AsyncReplayWorker
from locked_predictor import LockedPredictor


def test_async_replay_is_single_flight_and_one_shot():
    release = threading.Event()
    worker = AsyncReplayWorker("test")
    assert worker.submit(lambda: (release.wait(1.0), 7)[1])
    assert worker.running
    assert not worker.submit(lambda: 9)
    assert worker.poll() is None
    release.set()
    deadline = time.monotonic() + 1.0
    while worker.running and time.monotonic() < deadline:
        time.sleep(0.005)
    completion = worker.poll()
    assert completion is not None
    assert completion.value == 7
    assert completion.error is None
    assert worker.poll() is None


def test_async_replay_returns_error_without_raising_in_caller():
    worker = AsyncReplayWorker("error")

    def fail():
        raise RuntimeError("boom")

    assert worker.submit(fail)
    deadline = time.monotonic() + 1.0
    while worker.running and time.monotonic() < deadline:
        time.sleep(0.005)
    completion = worker.poll()
    assert isinstance(completion.error, RuntimeError)


def test_locked_predictor_serializes_calls():
    class Probe:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        def predict(self, value):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            self.active -= 1
            return value

    probe = Probe()
    predictor = LockedPredictor(probe)
    threads = [threading.Thread(target=predictor.predict, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert probe.max_active == 1
