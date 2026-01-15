"""Logging setup."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


def setup_logging(log_dir: str, logger_name: str = "edge_inference") -> logging.Logger:
    """Configure logging handlers and return the named logger."""
    log_path = os.path.join(
        log_dir, f"edge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(logger_name)
