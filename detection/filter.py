"""Filter detections based on label allow-lists."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from config.settings import INTERESTING_LABELS


class LabelFilter:
    """Filter detections to an allowed label set."""

    def __init__(self, labels: Optional[Iterable[str]] = None) -> None:
        label_set = labels if labels is not None else INTERESTING_LABELS
        self._labels = {normalize_label(label) for label in label_set if label}

    def filter(self, detections: List[Dict]) -> List[Dict]:
        return [
            detection
            for detection in detections
            if normalize_label(detection.get("class_name")) in self._labels
        ]


def normalize_label(name: Optional[str]) -> str:
    return (name or "").strip().lower()
