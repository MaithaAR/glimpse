"""Fix every random number generator so the same command gives the same numbers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. CPU runs are then repeatable; GPU kernels may still differ slightly."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.bind(stage="seed").info("random seed set to {}", seed)
