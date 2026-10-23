"""Training-time augmentation: a little random noise on real sensor readings, so the model learns the trend
rather than exact numbers. Applied to training batches only (validation, test and the demo never see it).
No time shift: it would flatten the last 30 minutes, which carry the strongest pre-meal signal."""

from __future__ import annotations

import numpy as np

from glimpse.config import AugmentCfg


def add_sensor_noise(
    glucose: np.ndarray, heart_rate: np.ndarray, cfg: AugmentCfg, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return noisy copies of raw windows (mg/dL, bpm). Empty slots stay empty, so gap filling and the
    missing-glucose flag are unchanged. With augmentation switched off, the inputs are returned unchanged."""
    if not cfg.enabled:
        return glucose, heart_rate
    g = glucose + rng.normal(0.0, cfg.glucose_noise_mgdl, glucose.shape)  # NaN + noise stays NaN
    h = heart_rate + rng.normal(0.0, cfg.hr_noise_bpm, heart_rate.shape)
    return g, h
