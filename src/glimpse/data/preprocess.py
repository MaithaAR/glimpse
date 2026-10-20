"""Sensor preprocessing shared by training and inference (one code path for both)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, ConfigDict

from glimpse.schemas import GlucoseReading

STEP_MIN = 5


class CGMWindow(BaseModel):
    """Pre-meal glucose resampled to a regular 5-minute grid ending at the meal time."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    values: np.ndarray  # shape (n_steps,), NaN where no reading
    coverage: float  # share of grid points with a reading
    max_gap_min: float  # longest stretch without a reading, including the window edges
    readings_last_30min: int


def cgm_window(readings: list[GlucoseReading], meal_time: datetime, lookback_min: int) -> CGMWindow:
    """Resample readings in [meal_time - lookback, meal_time] onto a 5-min grid (mean per bin, NaN if empty)."""
    n = lookback_min // STEP_MIN
    start = meal_time - timedelta(minutes=lookback_min)
    sums, counts = np.zeros(n), np.zeros(n)
    inside = [r for r in readings if start < r.timestamp <= meal_time]
    for r in inside:
        # bin k covers (start + 5k, start + 5(k+1)] so the reading at the meal time falls in the last bin
        idx = min(max(math.ceil((r.timestamp - start).total_seconds() / (60 * STEP_MIN)) - 1, 0), n - 1)
        sums[idx] += r.glucose_mgdl
        counts[idx] += 1
    values = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    edges = [start] + [r.timestamp for r in inside] + [meal_time]
    max_gap = max((b - a).total_seconds() / 60 for a, b in zip(edges[:-1], edges[1:], strict=True))
    last30 = sum(1 for r in inside if r.timestamp > meal_time - timedelta(minutes=30))
    return CGMWindow(values=values, coverage=float((counts > 0).mean()), max_gap_min=max_gap, readings_last_30min=last30)
