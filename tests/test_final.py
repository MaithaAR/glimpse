"""Decision thresholds: the cost rule picks sensible cut-offs, and frozen thresholds cannot be edited silently."""

import json

import numpy as np
import pytest

from glimpse.config import load_config
from glimpse.evaluation import choose_thresholds, expected_cost
from glimpse.training.final import cost_dict, hash_thresholds, load_thresholds


def test_thresholds_separate_clear_cases():
    """Clearly separated scores: everything low is cleared, everything high gets a caution, at zero cost."""
    p, y = np.array([0.05, 0.1, 0.15, 0.8, 0.9]), np.array([0, 0, 0, 1, 1])
    lo, hi, cost = choose_thresholds(p, y, cost_dict(load_config()))
    assert cost == 0.0 and 0.15 < lo <= hi <= 0.8


def test_chosen_thresholds_never_cost_more_than_simple_policies():
    """The chosen pair is at least as good as 'always low risk' or 'always caution' on the same meals."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 200), 0, 1)
    costs = cost_dict(load_config())
    _, _, cost = choose_thresholds(p, y, costs)
    assert cost <= min(expected_cost(np.zeros(200, int), y, costs), expected_cost(np.ones(200, int), y, costs))


def test_frozen_thresholds_detect_edits(tmp_path):
    """Changing a threshold after freezing makes loading fail."""
    d = {"low_risk_below": 0.1, "caution_from": 0.3}
    d["sha256"] = hash_thresholds(d)
    f = tmp_path / "t.json"
    f.write_text(json.dumps(d))
    assert load_thresholds(f)["caution_from"] == 0.3
    d["caution_from"] = 0.5
    f.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_thresholds(f)
