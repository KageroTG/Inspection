"""FPS tracking utility."""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Deque


class FpsTracker:
    """Track frames-per-second using a rolling time window."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=max(2, window_size))
        self._fps: float = 0.0

    def update(self) -> float:
        """Record a new frame timestamp and return the current FPS."""
        now = perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            duration = self._timestamps[-1] - self._timestamps[0]
            if duration > 0:
                self._fps = (len(self._timestamps) - 1) / duration
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps
