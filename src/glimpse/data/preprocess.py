"""Sensor preprocessing shared by training and inference (one code path for both)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, ConfigDict

from glimpse.schemas import GlucoseReading, HeartRateReading

STEP_MIN = 5


class CGMWindow(BaseModel):
    """Pre-meal signal resampled to a regular 5-minute grid ending at the meal time."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    values: np.ndarray  # shape (n_steps,), NaN where no reading
    coverage: float  # share of grid points with a reading
    max_gap_min: float  # longest stretch without a reading, including the window edges
    readings_last_30min: int


def resample_window(
    times: Sequence[datetime], values: Sequence[float], meal_time: datetime, lookback_min: int
) -> CGMWindow:
    """Average readings in (meal_time - lookback, meal_time] into 5-min bins; bin k covers (start+5k, start+5(k+1)]."""
    n = lookback_min // STEP_MIN
    start = meal_time - timedelta(minutes=lookback_min)
    sums, counts = np.zeros(n), np.zeros(n)
    inside = [(t, v) for t, v in zip(times, values, strict=True) if start < t <= meal_time and not math.isnan(v)]
    for t, v in inside:
        idx = min(max(math.ceil((t - start).total_seconds() / (60 * STEP_MIN)) - 1, 0), n - 1)
        sums[idx] += v
        counts[idx] += 1
    edges = [start] + [t for t, _ in inside] + [meal_time]
    max_gap = max((b - a).total_seconds() / 60 for a, b in zip(edges[:-1], edges[1:], strict=True))
    last30 = sum(1 for t, _ in inside if t > meal_time - timedelta(minutes=30))
    return CGMWindow(
        values=np.where(counts > 0, sums / np.maximum(counts, 1), np.nan),
        coverage=float((counts > 0).mean()),
        max_gap_min=max_gap,
        readings_last_30min=last30,
    )


def cgm_window(readings: list[GlucoseReading], meal_time: datetime, lookback_min: int) -> CGMWindow:
    """Glucose window from validated readings (used identically at training and inference time)."""
    return resample_window([r.timestamp for r in readings], [r.glucose_mgdl for r in readings], meal_time, lookback_min)


def hr_window(readings: list[HeartRateReading], meal_time: datetime, lookback_min: int) -> CGMWindow:
    """Heart-rate window on the same 5-min grid as glucose."""
    return resample_window([r.timestamp for r in readings], [r.bpm for r in readings], meal_time, lookback_min)


def fill_gaps(x: np.ndarray, fallback: float) -> np.ndarray:
    """Linearly interpolate interior NaNs, carry the nearest value to the edges, use `fallback` if all are NaN."""
    x = np.asarray(x, dtype=float).copy()
    ok = ~np.isnan(x)
    if not ok.any():
        return np.full_like(x, fallback)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])
