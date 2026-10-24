"""Sensor preprocessing shared by training and inference: the demo calls exactly these functions.

A window covers the `lookback_min` minutes BEFORE the meal, split into 5-minute slots. Slot k holds the average of
the readings in (start + 5k, start + 5(k+1)] minutes. Nothing after the meal time is ever read."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, ConfigDict

STEP_MIN = 5


class SensorWindow(BaseModel):
    """A pre-meal signal on the regular 5-minute grid, plus simple quality numbers."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    values: np.ndarray  # shape (lookback_min / 5,), NaN where a slot had no reading
    coverage: float  # share of slots with at least one reading
    max_gap_min: float  # longest stretch without a reading, counting the window edges
    readings_last_30min: int


def resample_window(times: Sequence[datetime], values: Sequence[float], meal_time: datetime,
                    lookback_min: int = 120) -> SensorWindow:  # fmt: skip
    """Average the readings of the `lookback_min` minutes before `meal_time` into 5-minute slots."""
    n = lookback_min // STEP_MIN
    start = meal_time - timedelta(minutes=lookback_min)
    sums, counts = np.zeros(n), np.zeros(n)
    # sorted by time, so the gaps below are measured between neighbouring readings even if input is unordered
    inside = sorted((t, v) for t, v in zip(times, values, strict=True)
                    if start < t <= meal_time and v is not None and not math.isnan(v))  # fmt: skip
    for t, v in inside:
        k = min(max(math.ceil((t - start).total_seconds() / (60 * STEP_MIN)) - 1, 0), n - 1)
        sums[k] += v
        counts[k] += 1
    edges = [start] + [t for t, _ in inside] + [meal_time]
    max_gap = max((b - a).total_seconds() / 60 for a, b in zip(edges[:-1], edges[1:], strict=True))
    last30 = sum(1 for t, _ in inside if t > meal_time - timedelta(minutes=30))
    return SensorWindow(values=np.where(counts > 0, sums / np.maximum(counts, 1), np.nan),
                        coverage=float((counts > 0).mean()), max_gap_min=max_gap,
                        readings_last_30min=last30)  # fmt: skip


def fill_gaps(x: np.ndarray, fallback: float) -> np.ndarray:
    """Fill empty slots: straight line between neighbouring readings, nearest reading at the edges,
    `fallback` (the training average) if the whole window is empty. Uses only values inside the window."""
    x = np.asarray(x, dtype=float).copy()
    ok = ~np.isnan(x)
    if not ok.any():
        return np.full_like(x, fallback)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def passes_premeal_checks(w: SensorWindow, min_last30: int, min_coverage: float, max_gap_min: float) -> bool:
    """The product's rule before any model runs: enough recent readings, enough coverage, no long gap."""
    return w.readings_last_30min >= min_last30 and w.coverage >= min_coverage and w.max_gap_min <= max_gap_min
