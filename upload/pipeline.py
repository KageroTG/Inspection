"""Upload pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, Optional

import cv2

from config.settings import (
    API_KEY,
    API_URL,
    AWS_ACCESS_KEY,
    AWS_BASE_URL,
    AWS_BUCKET,
    AWS_REGION,
    AWS_SECRET_KEY,
    CRACK_LABELS,
    CRACK_UPLOAD_DELAY,
    DETECTION_TYPE,
    IMMEDIATE_UPLOAD_LABELS,
    LOCATION,
)
from upload.api_client import ApiClient
from upload.batcher import CrackDebouncer
from upload.deduplicator import TrackDeduplicator
from upload.s3_uploader import S3Uploader


@dataclass
class UploadRecord:
    url: str
    timestamp: str
    label: str


class UploadPipeline:
    """Handle immediate and debounced uploads."""

    def __init__(
        self,
        logger,
        jpeg_quality: int = 85,
        dedup_size: int = 1000,
    ) -> None:
        self._logger = logger
        self._jpeg_quality = jpeg_quality
        self._deduplicator = TrackDeduplicator(dedup_size)
        self._tz = timezone(timedelta(hours=8))
        self.last_upload: Optional[UploadRecord] = None

        self._s3 = None
        self._api = None
        if AWS_ACCESS_KEY and AWS_SECRET_KEY and AWS_BUCKET:
            self._s3 = S3Uploader(
                AWS_ACCESS_KEY,
                AWS_SECRET_KEY,
                AWS_REGION,
                AWS_BUCKET,
                AWS_BASE_URL,
                logger=logger,
            )
        else:
            self._logger.warning("S3 credentials missing; uploads disabled")

        if API_URL and API_KEY:
            self._api = ApiClient(API_URL, API_KEY, logger=logger)
        else:
            self._logger.warning("API credentials missing; API posts disabled")

        self._crack_debouncer = CrackDebouncer(
            CRACK_UPLOAD_DELAY, self._upload_crack
        )

    def process_frame(
        self,
        frame,
        detections: Iterable[Dict],
        frame_index: int,
    ) -> None:
        if not detections:
            return
        if self._s3 is None:
            return

        image_bytes = self._encode_frame(frame)
        if image_bytes is None:
            return

        cracks = []
        immediate = []
        for detection in detections:
            label = _normalize_label(detection.get("class_name"))
            if label in CRACK_LABELS:
                cracks.append(detection)
            elif label in IMMEDIATE_UPLOAD_LABELS:
                immediate.append(detection)

        for detection in immediate:
            if not self._deduplicator.should_upload(detection):
                continue
            self._upload_immediate(image_bytes, detection, frame_index)

        if cracks:
            self._crack_debouncer.schedule(image_bytes, cracks[-1])

    def flush(self) -> None:
        self._crack_debouncer.flush()

    def _encode_frame(self, frame) -> Optional[bytes]:
        success, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not success:
            self._logger.error("Failed to encode frame for upload")
            return None
        return encoded.tobytes()

    def _upload_immediate(
        self,
        image_bytes: bytes,
        detection: Dict,
        frame_index: int,
    ) -> None:
        label = _normalize_label(detection.get("class_name"))
        now = datetime.now(self._tz)
        key = _build_key(label, frame_index, now)
        url = self._s3.upload_bytes(image_bytes, key)
        if not url:
            return
        payload = _build_payload(label, url, detection)
        if self._api:
            self._api.post(payload)
        self.last_upload = UploadRecord(url=url, timestamp=now.isoformat(), label=label)

    def _upload_crack(self, image_bytes: bytes, detection: Dict) -> None:
        label = _normalize_label(detection.get("class_name"))
        now = datetime.now(self._tz)
        key = _build_key(label, int(now.timestamp()), now)
        url = self._s3.upload_bytes(image_bytes, key)
        if not url:
            return
        payload = _build_payload(label, url, detection)
        if self._api:
            self._api.post(payload)
        self.last_upload = UploadRecord(url=url, timestamp=now.isoformat(), label=label)


def _build_payload(label: str, image_url: str, detection: Dict) -> Dict:
    detection_id = detection.get("track_id")
    if detection_id is None:
        detection_id = int(datetime.now().timestamp())
    return {
        "did": str(detection_id),
        "type": DETECTION_TYPE,
        "detect": label,
        "image": image_url,
        "location": LOCATION,
    }


def _build_key(label: str, frame_index: int, now: datetime) -> str:
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    suffix = now.microsecond
    return f"{label}-frame{frame_index}-{timestamp}-{suffix}.jpg"


def _normalize_label(name: Optional[str]) -> str:
    return (name or "").strip().lower()
