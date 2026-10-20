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
