"""Threaded video recorder using OpenCV VideoWriter."""

from __future__ import annotations

import logging
import threading
from queue import Empty, Full, Queue
from typing import Optional, Tuple

import cv2


class VideoRecorder:
    """Write frames to disk in a background thread."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        logger: Optional[logging.Logger] = None,
        queue_size: int = 120,
        fourcc: str = "mp4v",
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._queue: Queue = Queue(maxsize=max(1, queue_size))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._dropped = 0
        self._output_path = output_path

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            frame_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open VideoWriter for {output_path}")
        self._writer = writer

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame) -> None:
        if self._stop_event.is_set():
            return
        try:
            self._queue.put_nowait(frame)
        except Full:
            try:
                _ = self._queue.get_nowait()
                self._dropped += 1
            except Empty:
                return
            try:
                self._queue.put_nowait(frame)
            except Full:
                self._dropped += 1

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._writer.release()
        if self._dropped:
            self._logger.warning(
                "Recorder dropped %s frames due to queue backpressure",
                self._dropped,
            )
        self._logger.info("Recording saved to %s", self._output_path)

    def _worker(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._writer.write(frame)
            except Exception as exc:
                self._logger.warning("Failed to write frame: %s", exc)
