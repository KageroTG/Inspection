"""Track-based and spatial deduplication helpers."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Set, Tuple


class TrackDeduplicator:
    """Prevent repeated uploads for the same target."""

    def __init__(self, max_size: int = 1000, spatial_quant_px: int = 32) -> None:
        self._max_size = max(1, max_size)
        self._order: Deque[int] = deque()
        self._seen: Set[int] = set()
        self._spatial_order: Deque[Tuple] = deque()
        self._spatial_seen: Set[Tuple] = set()
        self._spatial_quant = max(1, spatial_quant_px)

    def should_upload(self, detection: dict) -> bool:
        track_id = detection.get("track_id")
        key = self._spatial_key(detection)

        if key is not None and key in self._spatial_seen:
            return False
        if track_id is not None and track_id in self._seen:
            return False

        if track_id is not None:
            self._track_add(track_id)
        if key is not None:
            self._spatial_add(key)
        return True

    def _track_add(self, track_id: int) -> None:
        self._seen.add(track_id)
        self._order.append(track_id)
        if len(self._order) > self._max_size:
            oldest = self._order.popleft()
            self._seen.discard(oldest)

    def _spatial_add(self, key: Tuple) -> None:
        self._spatial_seen.add(key)
        self._spatial_order.append(key)
        if len(self._spatial_order) > self._max_size:
            oldest = self._spatial_order.popleft()
            self._spatial_seen.discard(oldest)

    def _spatial_key(self, detection: dict) -> Optional[Tuple]:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = int(((x1 + x2) / 2) // self._spatial_quant)
        cy = int(((y1 + y2) / 2) // self._spatial_quant)
        w = int((x2 - x1) // self._spatial_quant)
        h = int((y2 - y1) // self._spatial_quant)
        label = _normalize_label(detection.get("class_name"))
        return (label, cx, cy, w, h)


def _normalize_label(name: Optional[str]) -> str:
    return (name or "").strip().lower()
