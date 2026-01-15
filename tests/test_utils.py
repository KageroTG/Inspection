import logging

import pytest

from utils.directories import ensure_directories
from utils.logger import setup_logging
from utils.validators import validate_environment


def _reset_root_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_ensure_directories_creates_paths(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    images_dir = tmp_path / "images"

    ensure_directories(str(logs_dir), str(images_dir))

    assert logs_dir.is_dir()
    assert images_dir.is_dir()


def test_setup_logging_creates_log_file(tmp_path) -> None:
    _reset_root_logging()
    logger = setup_logging(str(tmp_path))

    logger.info("test log line")

    log_files = list(tmp_path.glob("edge_*.log"))
    assert log_files, "expected a log file in the log directory"


def test_validate_environment_raises_on_missing() -> None:
    logger = logging.getLogger("test_validate")

    with pytest.raises(SystemExit):
        validate_environment("", None, None, "", logger)
