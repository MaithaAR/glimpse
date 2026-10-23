"""Photo model and photo preparation (random weights: no download needed, so this also runs on GitHub)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from glimpse.config import load_config
from glimpse.data.photos import PhotoError, open_photo, photo_tensor
from glimpse.models.photo_model import PhotoCarbModel, carb_bin, n_trainable

ARCH = "convnext_tiny.fb_in22k_ft_in1k"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return PhotoCarbModel(ARCH).eval()


def test_heads_shapes(model):
    """2 photos -> 2 log-carb values and 2 x 4 bin scores; the backbone summary has 768 numbers."""
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert model.embed(x).shape == (2, 768)
        reg, bins = model(x)
    assert reg.shape == (2,) and bins.shape == (2, 4)


def test_size_and_freezing(model):
    """About 28 million numbers in total; frozen -> only the heads learn; unfreezing adds the last stage."""
    total = sum(p.numel() for p in model.parameters())
    assert 27_000_000 < total < 29_000_000 < 200_000_000
    heads = 768 * 1 + 1 + 768 * 4 + 4
    model.set_trainable(last_stage=False)
    assert n_trainable(model) == heads
    model.set_trainable(last_stage=True)
    assert n_trainable(model) > 10_000_000
    model.set_trainable(last_stage=False)


def test_carb_bins_at_the_edges():
    """<30 -> 0, 30 -> 1, 59.99 -> 1, 60 -> 2, 90 -> 3."""
    g = torch.tensor([0.0, 29.99, 30.0, 59.99, 60.0, 89.99, 90.0, 250.0])
    assert carb_bin(g).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_photo_preparation(tmp_path):
    """Any usable photo becomes 3 x 224 x 224; a phone's rotation flag is applied; tiny or broken files are refused."""
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90 degrees" flag, as phones write it
    img.save(tmp_path / "rotated.jpg", exif=exif)
    assert open_photo(tmp_path / "rotated.jpg").size == (300, 400)  # width and height swapped back upright
    assert photo_tensor(tmp_path / "rotated.jpg", ARCH).shape == (3, 224, 224)
    Image.new("RGB", (40, 40)).save(tmp_path / "tiny.png")
    (tmp_path / "broken.jpg").write_bytes(b"not a photo")
    for bad in ("tiny.png", "broken.jpg"):
        with pytest.raises(PhotoError):
            open_photo(tmp_path / bad)


@pytest.mark.skipif(not any(Path("data/weights").glob("models--timm--convnext_tiny*")), reason="weights not downloaded")
def test_pinned_weights_load_offline():
    """The verified local weights load without network and change the backbone from its random start."""
    from glimpse.models.weights import weights_path

    cfg = load_config()
    path = weights_path(cfg.photo, download=False)
    m = PhotoCarbModel(ARCH, weights=path)
    assert not torch.equal(m.backbone.stem[0].weight, PhotoCarbModel(ARCH).backbone.stem[0].weight)
