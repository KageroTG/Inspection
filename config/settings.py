"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

LOG_DIR = "logs"
IMAGES_DIR = "images"
VIDEOS_DIR = "videos"

MODEL_PATH = os.getenv("MODEL_PATH", "best_8n_8.2.pt")
CAMERA_SOURCE_RAW = os.getenv("CAMERA_SOURCE", "Ipoh to KL - 15minutes.mp4")
CAMERA_WIDTH = os.getenv("CAMERA_WIDTH")
CAMERA_HEIGHT = os.getenv("CAMERA_HEIGHT")
RECONNECT_DELAY = float(os.getenv("CAMERA_RECONNECT_DELAY", "1.5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.4"))
UPLOAD_WORKERS = max(2, int(os.getenv("UPLOAD_WORKERS", "4")))
CRACK_UPLOAD_DELAY = float(os.getenv("CRACK_UPLOAD_DELAY", "5.0"))
PERF_LOG_INTERVAL = float(os.getenv("PERF_LOG_INTERVAL", "10.0"))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1") == "1"
RECORD = os.getenv("RECORD", "0") == "1"
RECORD_ALL_FRAMES = os.getenv("RECORD_ALL_FRAMES", "1") == "1"

# Detection configuration


LOCATION = os.getenv("ROBOLOCATION", "1111110000.1551331")
DETECTION_TYPE = os.getenv("ROBODETECTION_TYPE", "RoAd")


def _parse_label_set(raw: Optional[str], fallback: List[str]) -> set:
    if not raw:
        return {item.strip().lower() for item in fallback if item.strip()}
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


IMMEDIATE_UPLOAD_LABELS = _parse_label_set(
    os.getenv("PRIORITY_CLASSES"),
    ["potholes", "raveling", "stagnant_water"],
)
CRACK_LABELS = _parse_label_set(
    os.getenv("CRACK_CLASSES"),
    ["cracks"],
)
INTERESTING_LABELS = IMMEDIATE_UPLOAD_LABELS | CRACK_LABELS

# API / AWS configuration
API_URL = os.getenv("ROBOLYZE_API_URL")
API_KEY = os.getenv("ROBOLYZE_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET = os.getenv("AWS_BUCKET", "robolyzedatamy")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-5")
AWS_BASE_URL = os.getenv(
    "AWS_BASE_URL",
    f"https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com",
)

# GPU configuration
USE_GPU = os.getenv("USE_GPU", "1") == "1"

FPS_WINDOW = int(os.getenv("FPS_WINDOW", "30"))


def _parse_camera_source(raw: str):
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


CAMERA_SOURCE = _parse_camera_source(CAMERA_SOURCE_RAW)
CAMERA_WIDTH_VALUE = _parse_int(CAMERA_WIDTH)
CAMERA_HEIGHT_VALUE = _parse_int(CAMERA_HEIGHT)
