"""Evaluation helpers: cost-only cut-offs, person-level ranges, calibration and the status rule."""

import numpy as np

from glimpse.config import load_config
from glimpse.evaluation import cost_only_thresholds, person_bootstrap_ci, reliability
from glimpse.run_evaluation import _status
from glimpse.training.final import cost_dict


def test_cost_only_cutoffs_follow_from_the_costs():
    """With costs (5,0) / (0,1) / (2,0.5): low risk below 1/7, caution from 0.2, can't judge in between."""
    lo, hi = cost_only_thresholds(cost_dict(load_config()))
    assert np.isclose(lo, 0.5 / 3.5) and np.isclose(hi, 0.2)


def test_expensive_abstain_collapses_to_one_cutoff():
    """If 'can't judge' always costs more, there is a single cut-off where low risk and caution cost the same."""
    lo, hi = cost_only_thresholds({"low_risk": (5, 0), "caution": (0, 1), "abstain": (8, 2)})
    assert lo == hi == 1 / 6


def test_person_bootstrap_range_contains_the_estimate():
    """A mean over meals: the 95% range by person brackets the estimate."""
    rng = np.random.default_rng(0)
    sids, v = np.repeat(np.arange(10), 20), rng.normal(0, 1, 200)
    point, lo, hi = person_bootstrap_ci(sids, lambda i: float(v[i].mean()), B=300)
    assert lo <= point <= hi


def test_reliability_of_honest_probabilities_is_near_zero():
    """Labels drawn with the stated probabilities give a small ECE."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    _, ece = reliability(p, (rng.uniform(0, 1, 20000) < p).astype(int), 10)
    assert ece < 0.02


def test_status_rule():
    """met = whole range meets; on track = estimate only; otherwise at risk."""
    ok = lambda v: v > 0  # noqa: E731
    assert _status(0.05, 0.01, 0.09, ok) == "met"
    assert _status(0.05, -0.01, 0.09, ok) == "on track"
    assert _status(-0.01, -0.05, 0.02, ok) == "at risk"
