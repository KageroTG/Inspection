"""API client with basic retry handling."""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import requests


class ApiClient:
    """POST detection payloads to the backend API."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        logger: Optional[logging.Logger] = None,
        retries: int = 3,
        timeout: float = 6.0,
    ) -> None:
        self._api_url = api_url
        self._api_key = api_key
        self._logger = logger or logging.getLogger(__name__)
        self._retries = max(1, retries)
        self._timeout = timeout

    def post(self, payload: Dict) -> bool:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self._api_key,
        }
        for attempt in range(1, self._retries + 1):
            try:
                response = requests.post(
                    self._api_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
                if response.status_code in (200, 201):
                    self._logger.info(
                        "API_POST success attempt=%s status=%s",
                        attempt,
                        response.status_code,
                    )
                    return True
                self._logger.warning(
                    "API_POST status=%s attempt=%s",
                    response.status_code,
                    attempt,
                )
            except Exception as exc:
                self._logger.warning(
                    "API_POST failed attempt=%s error=%s", attempt, exc
                )
                time.sleep(1)
        return False
