"""Seeds give repeatable numbers; logging writes a file; the command checks the settings."""

import numpy as np
import torch
from loguru import logger

from glimpse.cli import main
from glimpse.utils.logging import setup_logging
from glimpse.utils.seed import set_seed


def test_same_seed_same_numbers():
    """Two runs with the same seed draw identical NumPy and PyTorch numbers."""
    set_seed(7)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(7)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_log_file_is_written(tmp_path):
    """A logged message reaches logs/glimpse.log, with its context tag."""
    path = setup_logging(tmp_path)
    logger.bind(stage="test").info("hello")
    logger.complete()
    text = path.read_text()
    assert "hello" in text and "'stage': 'test'" in text


def test_check_config_command(tmp_path):
    """`glimpse check-config` exits 0 on the real file and 2 on a broken one."""
    assert main(["check-config"]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: -1\n")
    assert main(["--config", str(bad), "check-config"]) == 2


def test_params_command_runs():
    """`glimpse params` is wired up and writes the budget table (catches a missing command entry)."""
    assert main(["params"]) == 0
