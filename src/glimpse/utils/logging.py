"""Loguru setup: human-readable stderr plus a rotating log file under logs/."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """Configure Loguru once: stderr and logs/glimpse.log (rotated at 5 MB, 10 files kept)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {extra} | {message}"
    logger.configure(extra={"stage": "-"})
    logger.add(sys.stderr, level=level, format=fmt)
    logger.add(
        Path(log_dir) / "glimpse.log",
        level="DEBUG",
        rotation="5 MB",
        retention=10,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {extra} | {name}:{function}:{line} | {message}",
    )
    _CONFIGURED = True
