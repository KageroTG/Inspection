"""Environment validation helpers."""

from __future__ import annotations

import logging


def validate_environment(
    api_url: str,
    api_key: str,
    aws_access_key: str,
    aws_secret_key: str,
    logger: logging.Logger,
) -> None:
    """Validate required environment variables and exit on failure."""
    missing = []
    if not api_url:
        missing.append("ROBOLYZE_API_URL")
    if not api_key:
        missing.append("ROBOLYZE_API_KEY")
    if not aws_access_key:
        missing.append("AWS_ACCESS_KEY_ID")
    if not aws_secret_key:
        missing.append("AWS_SECRET_ACCESS_KEY")

    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        raise SystemExit(1)
