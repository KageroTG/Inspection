"""S3 upload helpers."""

from __future__ import annotations

import logging
from typing import Optional

import boto3


class S3Uploader:
    """Upload JPEG bytes to S3."""

    def __init__(
        self,
        aws_access_key: str,
        aws_secret_key: str,
        aws_region: str,
        aws_bucket: str,
        aws_base_url: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._bucket = aws_bucket
        self._base_url = aws_base_url.rstrip("/")
        self._client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region,
        )

    def upload_bytes(self, image_bytes: bytes, key: str) -> Optional[str]:
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=image_bytes,
                ContentType="image/jpeg",
            )
            return f"{self._base_url}/{key}"
        except Exception as exc:
            self._logger.error("S3 upload failed key=%s error=%s", key, exc)
            return None
