"""The shared window function: right shape, no look-ahead, sensible gap filling and checks."""

from datetime import datetime, timedelta

import numpy as np

from glimpse.data.preprocess import fill_gaps, passes_premeal_checks, resample_window

MEAL = datetime(2020, 5, 1, 12, 0)


def _readings(every_min=5, start_min=-120, end_min=120, value=lambda m: 100 + m / 10):
    """Synthetic CGM readings from start_min to end_min around the meal (minutes)."""
    ts = [MEAL + timedelta(minutes=m) for m in range(start_min, end_min + 1, every_min)]
    return ts, [value((t - MEAL).total_seconds() / 60) for t in ts]


def test_window_shape_and_full_coverage():
    """2 h before the meal -> 24 five-minute slots, all filled when a reading arrives every 5 min."""
    ts, vs = _readings()
    w = resample_window(ts, vs, MEAL, 120)
    assert w.values.shape == (24,) and w.coverage == 1.0 and w.readings_last_30min == 6


def test_no_look_ahead():
    """Changing every reading AFTER the meal leaves the model input exactly the same."""
    ts, vs = _readings()
    changed = [v if t <= MEAL else v + 500 for t, v in zip(ts, vs, strict=True)]
    a = resample_window(ts, vs, MEAL, 120).values
    b = resample_window(ts, changed, MEAL, 120).values
    assert np.array_equal(a, b)


def test_gap_is_filled_with_a_straight_line():
    """An empty slot between 100 and 110 becomes 105; empty edges take the nearest reading."""
    x = np.array([np.nan, 100.0, np.nan, 110.0, np.nan])
    assert np.allclose(fill_gaps(x, fallback=120.0), [100, 100, 105, 110, 110])
    assert np.allclose(fill_gaps(np.full(3, np.nan), fallback=120.0), 120.0)


def test_sparse_window_fails_the_checks():
    """Readings only every 30 min: too few recent readings and gaps too long -> not usable before a meal."""
    ts, vs = _readings(every_min=30)
    w = resample_window(ts, vs, MEAL, 120)
    assert not passes_premeal_checks(w, min_last30=3, min_coverage=0.7, max_gap_min=20)
    ts, vs = _readings(every_min=5)
    assert passes_premeal_checks(resample_window(ts, vs, MEAL, 120), 3, 0.7, 20)
