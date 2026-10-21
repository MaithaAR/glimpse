"""Tests for evaluation utilities."""

import numpy as np

from glimpse.evaluation.metrics import ece_equal_mass, expected_cost, load_locked_set, noninferiority


def test_noninferiority():
    """Decision rule boundaries."""
    assert noninferiority(-0.01, 0.02) == "non-inferior"
    assert noninferiority(-0.09, -0.04) == "inferior"
    assert noninferiority(-0.05, 0.01) == "inconclusive"


def test_cost():
    """Expected cost uses (spike, no spike) columns."""
    costs = {"low_risk": [5, 0], "caution": [0, 1], "abstain": [2, 0.5]}
    y = np.array([1, 0])
    assert expected_cost(np.array([0, 0]), y, costs) == 2.5
    assert expected_cost(np.array([1, 1]), y, costs) == 0.5


def test_ece_perfect():
    """A calibrated predictor has small ECE."""
    p = np.linspace(0.05, 0.95, 1000)
    y = (np.random.default_rng(0).random(1000) < p).astype(int)
    assert ece_equal_mass(p, y) < 0.06


def test_locked_hash():
    """The locked final set is intact (hash verified on load)."""
    assert len(load_locked_set("splits/locked_final_set.json")) == 8


def test_ece_equal_width_bins_and_counts():
    """Equal-width ECE reports one row per bin with counts summing to n; a constant at the base rate has ECE 0."""
    from glimpse.evaluation.metrics import ece_equal_width

    y = np.array([1, 0, 0, 0] * 25)
    ece, table = ece_equal_width(np.full(100, 0.25), y, bins=10)
    assert len(table) == 10 and sum(t["n"] for t in table) == 100 and ece < 1e-9


def test_person_bootstrap_ci_contains_estimate():
    """The person-level CI brackets the full-sample statistic."""
    from glimpse.evaluation.metrics import person_bootstrap

    rng = np.random.default_rng(0)
    sids = np.repeat(np.arange(10), 20)
    x = rng.normal(size=200)
    lo, hi = person_bootstrap(np.mean, (x,), sids, B=500)
    assert lo < x.mean() < hi


def test_select_band_respects_safety_cap():
    """Meals cleared as low risk never exceed the allowed spike rate on the selection data."""
    from glimpse.evaluation.metrics import decide, select_band

    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(int)
    costs = {"low_risk": [3, 0], "caution": [0, 1], "abstain": [2, 0.5]}
    lo, hi = select_band(p, y, costs, max_abstain=0.2, max_missed_among_cleared=0.1)
    a = decide(p, lo, hi)
    assert lo <= hi and (a == 2).mean() <= 0.2
    assert y[a == 0].mean() <= 0.1


def test_engineered_features_shape():
    """Seven trend features plus the profile inputs, one row per meal."""
    from glimpse.data.dataset import TABULAR
    from glimpse.evaluation.baselines import SENSOR_FEATURES, engineered_features

    class _DS:
        glucose = np.tile(np.linspace(90, 120, 24), (3, 1))
        heart_rate = np.full((3, 24), 70.0)
        tabular = np.zeros((3, len(TABULAR)))

    x = engineered_features(_DS(), np.arange(3))
    assert x.shape == (3, len(SENSOR_FEATURES) + len(TABULAR))
    assert np.isclose(x[0, 3], 30 / 115, rtol=0.05)  # 30 mg/dL over 115 min -> slope per min
