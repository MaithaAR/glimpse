"""Tests for the evaluation helpers used by the proposal pilot."""

import numpy as np

from glimpse.evaluation import ece_equal_mass, expected_cost, load_locked_set, noninferiority


def test_noninferiority():
    assert noninferiority(-0.01, 0.02) == "non-inferior"
    assert noninferiority(-0.09, -0.04) == "inferior"
    assert noninferiority(-0.05, 0.01) == "inconclusive"


def test_cost():
    costs = {"low_risk": [5, 0], "caution": [0, 1], "abstain": [2, 0.5]}
    y = np.array([1, 0])
    assert expected_cost(np.array([0, 0]), y, costs) == 2.5
    assert expected_cost(np.array([1, 1]), y, costs) == 0.5


def test_ece_perfect():
    p = np.linspace(0.05, 0.95, 1000)
    y = (np.random.default_rng(0).random(1000) < p).astype(int)
    assert ece_equal_mass(p, y) < 0.06


def test_locked_hash():
    assert len(load_locked_set("splits/locked_final_set.json")) == 8


def test_calibration_by_person_finds_rare_spikers_over_predicted():
    """A model that predicts the same risk for everyone over-predicts for rare spikers and under-predicts for others."""
    from glimpse.evaluation import calibration_by_person

    person = np.repeat([1, 2, 3, 4], 10)
    y = np.concatenate([np.zeros(10), np.r_[np.ones(2), np.zeros(8)], np.r_[np.ones(8), np.zeros(2)], np.ones(10)])
    out = calibration_by_person(person, y, np.full(40, 0.5))
    assert out["rare_spikers"]["predicted"] > out["rare_spikers"]["observed"]
    assert out["frequent_spikers"]["predicted"] < out["frequent_spikers"]["observed"]
    assert out["corr_spike_rate_vs_gap"] < -0.9
