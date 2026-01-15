"""Camera input management."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Union

import cv2

CameraSource = Union[int, str]


class CameraManager:
    """OpenCV camera wrapper with basic reconnect handling."""

    def __init__(
        self,
        source: CameraSource,
        width: Optional[int] = None,
        height: Optional[int] = None,
        reconnect_delay: float = 1.5,
        logger: Optional[logging.Logger] = None,
        use_ffmpeg: bool = True,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self._logger = logger or logging.getLogger(__name__)
        self._use_ffmpeg = use_ffmpeg
        self._finished = False
        self._is_rtsp = isinstance(source, str) and source.lower().startswith(
            ("rtsp://", "rtsps://")
        )
        self._is_video_file = False
        if isinstance(source, str) and not self._is_rtsp:
            expanded = os.path.expanduser(source)
            abs_path = os.path.abspath(expanded)
            if os.path.isfile(expanded):
                self.source = expanded
                self._is_video_file = True
            elif os.path.isfile(abs_path):
                self.source = abs_path
                self._is_video_file = True
            else:
                self._is_video_file = True
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        if self.cap:
            self.cap.release()
        self._finished = False

        if self._is_rtsp and self._use_ffmpeg and hasattr(cv2, "CAP_FFMPEG"):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.source}")

        self._logger.info(
            "Stream ready source=%s resolution=%sx%s",
            self.source,
            self.width or "auto",
            self.height or "auto",
        )

    def read_frame(self):
        if self.cap is None:
            self._open()
        if self._finished:
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame

        if self._is_video_file:
            self._logger.info("Video source %s finished", self.source)
            self._finished = True
            return None

        self._logger.warning(
            "Camera frame missing. Reconnecting in %.1fs", self.reconnect_delay
        )
        time.sleep(self.reconnect_delay)
        try:
            self._open()
        except Exception as exc:
            self._logger.error("Failed to reopen camera: %s", exc)
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def read(self):
        return self.read_frame()

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
            self._logger.info("Camera stream released")

    def is_finished(self) -> bool:
        return self._finished
