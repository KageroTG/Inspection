"""Directory helpers."""

from __future__ import annotations

import os


def ensure_directories(*paths: str) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        if path:
            os.makedirs(path, exist_ok=True)
