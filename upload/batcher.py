"""Debounce logic for crack uploads."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Tuple


class CrackDebouncer:
    """Keep only the most recent crack detection within a time window."""

    def __init__(
        self,
        delay_seconds: float,
        callback: Callable[[bytes, dict], None],
    ) -> None:
        self.delay_seconds = delay_seconds
        self._callback = callback
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._payload: Optional[Tuple[bytes, dict]] = None
        self.last_scheduled_at: Optional[float] = None

    def schedule(self, image_bytes: bytes, detection: dict) -> None:
        with self._lock:
            self._payload = (image_bytes, detection.copy())
            self.last_scheduled_at = time.time()
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay_seconds, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            payload = self._payload
            self._payload = None
            self._timer = None
        if payload:
            image_bytes, detection = payload
            self._callback(image_bytes, detection)

    def flush(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            payload = self._payload
            self._payload = None
        if payload:
            image_bytes, detection = payload
            self._callback(image_bytes, detection)
