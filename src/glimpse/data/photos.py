"""Photo preparation shared by training, evaluation and the demo: open the image upright, check it is usable, then
apply the pretrained model's own resizing and colour scaling (taken from timm, not typed by hand)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError


class PhotoError(ValueError):
    """The photo cannot be used (unreadable, or too small)."""


def open_photo(path: str | Path, min_side_px: int = 64) -> Image.Image:
    """Open a photo the right way up (phones store rotation separately, as EXIF) in RGB; reject unusable ones."""
    try:
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
    except (OSError, UnidentifiedImageError) as e:
        raise PhotoError(f"cannot read photo {path}: {e}") from e
    if min(img.size) < min_side_px:
        raise PhotoError(f"photo {path} is too small ({img.size[0]}x{img.size[1]} px; minimum {min_side_px})")
    return img


@lru_cache(maxsize=4)
def eval_transform(arch: str, crop_pct: float = 1.0):
    """The backbone's own test-time preprocessing (input size, interpolation, mean/std from timm's model card).
    crop_pct = 1.0 keeps as much of the plate as possible instead of zooming in."""
    import timm
    from timm.data import create_transform, resolve_model_data_config

    cfg = resolve_model_data_config(timm.create_model(arch, pretrained=False, num_classes=0))
    cfg["crop_pct"] = crop_pct
    return create_transform(**cfg, is_training=False)


def photo_tensor(path: str | Path, arch: str, crop_pct: float = 1.0, min_side_px: int = 64) -> torch.Tensor:
    """Photo file -> model-ready tensor (3, 224, 224). The same function serves training and the demo."""
    return eval_transform(arch, crop_pct)(open_photo(path, min_side_px))


@lru_cache(maxsize=4)
def train_transform(arch: str):
    """Training-only augmentation (used only when the backbone's last stage learns): horizontal flip, brightness and
    contrast +/-10% (no hue change: colour tells foods apart), and crops keeping at least 80% of the photo."""
    import timm
    from timm.data import create_transform, resolve_model_data_config

    cfg = resolve_model_data_config(timm.create_model(arch, pretrained=False, num_classes=0))
    return create_transform(**cfg, is_training=True, scale=(0.8, 1.0), hflip=0.5, color_jitter=(0.1, 0.1, 0.0),
                            auto_augment=None, re_prob=0.0)  # fmt: skip
