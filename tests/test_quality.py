"""Augmentation is small and leaves gaps alone; the photo fingerprint finds copies; the ledger enforces the cap."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from glimpse.config import AugmentCfg
from glimpse.data.augment import add_sensor_noise
from glimpse.data.quality import dhash

HAVE_DATA = Path("data/dataset.json").is_file() and Path("splits/dev_splits.json").is_file()


def test_noise_is_small_and_keeps_gaps():
    """Same shape, empty slots stay empty, typical change about 3 mg/dL; switched off = unchanged."""
    rng = np.random.default_rng(0)
    g = np.full((200, 24), 110.0)
    g[:, 5] = np.nan
    h = np.full((200, 24), 75.0)
    on = AugmentCfg(enabled=True, glucose_noise_mgdl=3.0, hr_noise_bpm=1.0)
    g2, h2 = add_sensor_noise(g, h, on, rng)
    assert g2.shape == g.shape and np.isnan(g2[:, 5]).all()
    assert 2.5 < np.nanstd(g2 - g) < 3.5 and 0.8 < np.std(h2 - h) < 1.2
    off = on.model_copy(update={"enabled": False})
    g3, _ = add_sensor_noise(g, h, off, rng)
    assert g3 is g


def test_dhash_finds_resized_copy(tmp_path):
    """A resized copy has (almost) the same fingerprint; an unrelated image does not."""
    rng = np.random.default_rng(1)
    a = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    a.save(tmp_path / "a.png")
    a.resize((128, 128)).save(tmp_path / "a_big.png")
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "b.png")
    d = lambda x, y: (dhash(tmp_path / x) ^ dhash(tmp_path / y)).bit_count()  # noqa: E731
    assert d("a.png", "a.png") == 0 and d("a.png", "a_big.png") <= 4 and d("a.png", "b.png") > 10


@pytest.mark.skipif(not HAVE_DATA, reason="run `just prepare` first")
def test_ledger_respects_the_cap():
    """The worst-case photo count is under 10,000, and a lower cap stops the run."""
    from glimpse.config import load_config
    from glimpse.data.dataset import MealDataset
    from glimpse.data.quality import finetune_ledger
    from glimpse.data.splits import Splits

    cfg = load_config()
    ds, sp, root = (
        MealDataset.load(Path("data/dataset")),
        Splits.load(Path("splits/dev_splits.json")),
        Path(cfg.data.cgmacros_root),
    )
    t = finetune_ledger(ds, sp, root, cfg.finetune_cap_items)
    assert 0 < t["items"].iloc[-1] <= 10_000
    with pytest.raises(ValueError):
        finetune_ledger(ds, sp, root, cap=10)
