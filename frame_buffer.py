"""CPU-side pose sequence buffer shared by live temporal inference."""

from collections import deque
import time
from typing import Optional

import numpy as np


class FrameBuffer:
    """Circular buffer of COCO-17 XY, confidence, and capture timestamps.

    The deployed 6-class Keras classifier still consumes one ``(17, 2)``
    skeleton. This buffer only collects temporal context; a future TCN must
    consume the sequence methods explicitly.
    """

    def __init__(self, max_frames: int = 30):
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.max_frames = max_frames
        self.keypoints = deque(maxlen=max_frames)
        self.confidences = deque(maxlen=max_frames)
        self.timestamps = deque(maxlen=max_frames)
        self.last_update_time = time.time()

    def push(
        self,
        keypoints: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        *,
        timestamp: Optional[float] = None,
    ) -> None:
        """Append one pose sample without mixing XY and confidence."""
        xy = np.asarray(keypoints, dtype=np.float32)
        if xy.shape != (17, 2):
            raise ValueError(f"keypoints must have shape (17, 2), got {xy.shape}")

        if confidences is None:
            conf = np.ones(17, dtype=np.float32)
        else:
            conf = np.asarray(confidences, dtype=np.float32)
            if conf.shape != (17,):
                raise ValueError(
                    f"confidences must have shape (17,), got {conf.shape}"
                )

        now = time.time() if timestamp is None else float(timestamp)
        self.keypoints.append(xy.copy())
        self.confidences.append(conf.copy())
        self.timestamps.append(now)
        self.last_update_time = now

    def is_ready(self, required_frames: Optional[int] = None) -> bool:
        """Return whether the requested amount of temporal context exists."""
        threshold = max(1, self.max_frames // 2) if required_frames is None else int(required_frames)
        if threshold <= 0 or threshold > self.max_frames:
            raise ValueError("required_frames must be between 1 and max_frames")
        return len(self.keypoints) >= threshold

    def is_full(self) -> bool:
        return len(self.keypoints) == self.max_frames

    def get_latest_keypoints(self) -> Optional[np.ndarray]:
        if self.keypoints:
            return self.keypoints[-1]
        return None

    def get_all_keypoints(self) -> np.ndarray:
        """Return chronological XY sequence with shape ``(T, 17, 2)``."""
        if not self.keypoints:
            return np.empty((0, 17, 2), dtype=np.float32)
        return np.stack(self.keypoints)

    def get_all_confidences(self) -> np.ndarray:
        """Return chronological confidence sequence with shape ``(T, 17)``."""
        if not self.confidences:
            return np.empty((0, 17), dtype=np.float32)
        return np.stack(self.confidences)

    def get_timestamps(self) -> np.ndarray:
        return np.asarray(self.timestamps, dtype=np.float64)

    def clear(self) -> None:
        self.keypoints.clear()
        self.confidences.clear()
        self.timestamps.clear()

    def age_seconds(self) -> float:
        return time.time() - self.last_update_time
