import logging
import os
import time

import pytest
from config.settings import CAMERA_SOURCE, CAMERA_SOURCE_RAW
from core.camera import CameraManager

cv2 = pytest.importorskip("cv2")


def _source_kind(raw: str):
    if not raw:
        return None
    lower = raw.lower()
    if lower.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    if raw.isdigit():
        return "usb"
    return "local"


def _read_one_frame(source):
    logger = logging.getLogger("test_camera")
    camera = CameraManager(source, reconnect_delay=0.2, logger=logger)
    frame = None
    for _ in range(5):
        frame = camera.read_frame()
        if frame is not None:
            break
        time.sleep(0.1)
    camera.release()
    return frame


def test_camera_source_reads_frame():
    kind = _source_kind(CAMERA_SOURCE_RAW)
    if kind is None:
        pytest.skip("CAMERA_SOURCE is not configured")
    if kind == "local":
        assert os.path.isfile(CAMERA_SOURCE_RAW), "Local camera source file not found"

    frame = _read_one_frame(CAMERA_SOURCE)
    assert frame is not None, f"{kind} source returned no frame"
