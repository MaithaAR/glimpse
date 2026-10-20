"""Training-only augmentation for both modalities (never applied at validation, test or inference)."""

from __future__ import annotations

import numpy as np


def augment_sequence(
    seq: np.ndarray, rng: np.random.Generator, noise_sd: float = 0.05, max_shift: int = 1
) -> np.ndarray:
    """Add small Gaussian noise to the glucose and heart-rate channels and shift the window by up to one 5-min step.

    Sensor noise (~2-3 mg/dL at the dataset's glucose spread) and a +/-5-min timestamp jitter are realistic for
    consumer CGMs and phone-logged meal times. The mask channel is shifted but never noised.
    """
    out = seq.copy()
    out[..., :2, :] += rng.normal(0, noise_sd, size=out[..., :2, :].shape)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift:
        out = np.roll(out, shift, axis=-1)
        if shift > 0:
            out[..., :shift] = out[..., shift : shift + 1]  # repeat the first real step instead of wrapping
        else:
            out[..., shift:] = out[..., shift - 1 : shift]  # repeat the last real step
    return out.astype(seq.dtype)


def photo_transforms(model_name: str, train: bool):
    """timm preprocessing for the photo model: random-resized crop, flip and colour jitter when training,
    resize + centre crop otherwise; both use the backbone's own normalisation."""
    import timm
    from timm.data import create_transform, resolve_model_data_config

    cfg = resolve_model_data_config(timm.create_model(model_name, pretrained=False, num_classes=0))
    if train:
        return create_transform(**cfg, is_training=True, hflip=0.5, color_jitter=0.2, scale=(0.6, 1.0))
    return create_transform(**cfg, is_training=False)
