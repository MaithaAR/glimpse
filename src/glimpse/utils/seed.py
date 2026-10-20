"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
from loguru import logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and any accelerator) and request deterministic torch kernels."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.bind(stage="setup").info("seed set to {}", seed)
