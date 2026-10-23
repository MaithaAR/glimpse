"""Loguru setup: every message goes to the screen (stderr) and to a rotating file in logs/."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

FORMAT = "{time:HH:mm:ss} | {level:<7} | {extra} | {message}"


def setup_logging(log_dir: str | Path = "logs", level: str = "INFO") -> Path:
    """Send logs to stderr and to logs/glimpse.log (new file after 5 MB, last 5 files kept). Returns the file path."""
    path = Path(log_dir) / "glimpse.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()  # drop Loguru's default handler so messages are not printed twice
    logger.add(sys.stderr, level=level, format=FORMAT)
    file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {extra} | {name}:{function}:{line} | {message}"
    logger.add(path, level="DEBUG", rotation="5 MB", retention=5, encoding="utf-8", format=file_format)
    return path
