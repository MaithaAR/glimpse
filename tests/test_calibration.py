"""Fix 1 math (design D3 as amended): the penalised intercept, the deployment-order replay with the
two-hour label gate, and the joint (m, lo, hi) selection with its pre-registered tie-break."""

import numpy as np
import pytest

from glimpse.calibration import (
    M_GRID,
    choose_m_and_cutoffs,
    personal_offset,
    personal_risks,
    replay_offsets,
)

T0 = np.datetime64("2024-06-01T08:00")
H = np.timedelta64(1, "h")


def test_empty_history_and_infinite_m_give_zero():
    assert personal_offset(np.array([]), np.array([]), 2.0) == 0.0
    assert personal_offset(np.array([0.3, -1.2]), np.array([1, 0]), float("inf")) == 0.0


def test_same_labels_bounded_and_monotone_in_m():
    """The pre-registered reason m is mandatory: identical labels diverge unpenalised; with the prior the
    offset stays finite and shrinks as m grows."""
    f, y = np.full(5, 0.0), np.ones(5)
    bs = [personal_offset(f, y, m) for m in (0.5, 2.0, 8.0, 32.0)]
    assert all(np.isfinite(b) and 0 < b < 10 for b in bs)
    assert bs == sorted(bs, reverse=True)  # more shrinkage -> smaller offset


def test_newton_matches_brute_force_and_gradient_vanishes():
    rng = np.random.default_rng(0)
    f, y, m = rng.normal(0, 1.5, 7), (rng.uniform(size=7) < 0.4).astype(float), 3.0
    b = personal_offset(f, y, m)
    obj = lambda g: float(-(y * (f + g) - np.log1p(np.exp(f + g))).sum() + (m / 2) * g * g)  # noqa: E731
    grid = np.linspace(-5, 5, 20001)
    brute = float(grid[np.argmin([obj(g) for g in grid])])
    assert abs(b - brute) < 1e-3
    p = 1 / (1 + np.exp(-(f + b)))
    assert abs(float(np.sum(p - y)) + m * b) < 1e-6  # gradient ~ 0 at the optimum


def test_sign_symmetry():
    rng = np.random.default_rng(1)
    f, y = rng.normal(size=6), (rng.uniform(size=6) < 0.5).astype(float)
    assert abs(personal_offset(f, y, 2.0) + personal_offset(-f, 1 - y, 2.0)) < 1e-8


def test_replay_respects_time_order_and_two_hour_gate():
    """Meal 3 may use meal 1 (3 h earlier: label available) but not meal 2 (1 h earlier: window open)."""
    f, y = np.array([2.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0])
    times = np.array([T0, T0 + 2 * H, T0 + 3 * H])
    b = replay_offsets(f, y, times, m=1.0)
    only_first = personal_offset(f[:1], y[:1], 1.0)
    assert b[0] == 0.0  # first meal: no history
    assert abs(b[1] - only_first) < 1e-12  # meal 1's window closes exactly at t2: included (boundary)
    assert abs(b[2] - only_first) < 1e-12  # meal 2 (1 h before) is gated out; only meal 1 counts


def test_replay_ignores_later_meals_and_nonfinite_logits():
    f, y = np.array([0.5, np.nan, 0.5, -3.0]), np.array([1.0, 1.0, 0.0, 1.0])
    times = np.array([T0, T0 + 3 * H, T0 + 6 * H, T0 + 9 * H])
    b = replay_offsets(f, y, times, m=2.0)
    assert abs(b[2] - personal_offset(f[[0]], y[[0]], 2.0)) < 1e-12  # the NaN-logit meal never enters
    b2 = replay_offsets(f[:3], y[:3], times[:3], m=2.0)
    assert np.allclose(b[:3], b2)  # the last meal changes nothing before it


def test_infinite_m_is_exactly_the_population_arm():
    rng = np.random.default_rng(2)
    f = rng.normal(size=8)
    y, person = (rng.uniform(size=8) < 0.5).astype(float), np.array([1] * 4 + [2] * 4)
    times = np.array([T0 + 3 * k * H for k in range(4)] * 2)
    r = personal_risks(f, y, person, times, float("inf"))
    assert np.allclose(r, 1 / (1 + np.exp(-f)))


def test_tiebreak_prefers_the_largest_m():
    """One meal per person: every history is empty, every m gives identical risks and costs -> inf wins."""
    p = np.array([0.1, 0.3, 0.6, 0.8])
    y, person = np.array([0, 0, 1, 1]), np.arange(4)
    times = np.array([T0, T0, T0, T0])
    costs = {"low_risk": (5, 0), "caution": (0, 1), "abstain": (2, 0.5)}
    res = choose_m_and_cutoffs(p, y, person, times, costs)
    assert res["m"] == float("inf") and len(res["curve"]) == len(M_GRID)
    assert res["low_risk_below"] <= res["caution_from"]
