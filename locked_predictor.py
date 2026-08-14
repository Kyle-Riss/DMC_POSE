"""Serialize access to prediction backends that are shared by camera threads."""

from threading import Lock


class LockedPredictor:
    def __init__(self, predictor):
        self._predictor = predictor
        self._lock = Lock()

    def predict(self, *args, **kwargs):
        with self._lock:
            return self._predictor.predict(*args, **kwargs)

