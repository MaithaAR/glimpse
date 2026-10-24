"""The app's rules: profile order, decision words, glucose checks before the model, unfamiliar-photo distance."""

import math
from datetime import datetime, timedelta

import numpy as np
import torch

from glimpse.config import load_config
from glimpse.data.dataset import PROFILE
from glimpse.data.preprocess import resample_window
from glimpse.demo import decision, knn_distance, leave_one_out_distances, profile_vector, sentence, window_problem

MEAL = datetime(2020, 5, 1, 12, 0)


def _window(minutes_back: list[int], value: float = 100.0):
    times = [MEAL - timedelta(minutes=m) for m in minutes_back]
    return resample_window(times, [value] * len(times), MEAL, 120)


def test_profile_vector_order_and_meal_hour():
    """Same order as training; noon sits at the bottom of the 24-hour circle; unknown prior stays NaN."""
    v = profile_vector({"age": 50, "bmi": 30, "a1c_pct": 5.9, "fasting_glucose_mgdl": 105,
                        "prior_spike_rate": float("nan"), "prior_rise": 40}, MEAL)  # fmt: skip
    assert len(v) == len(PROFILE) and v[0] == 50 and np.isnan(v[PROFILE.index("prior_spike_rate")])
    assert math.isclose(v[PROFILE.index("hour_cos")], -1.0, abs_tol=1e-9)


def test_decision_and_words():
    assert decision(0.10, 0.143, 0.2) == "low risk"
    assert decision(0.17, 0.143, 0.2) == "can't judge"
    assert decision(0.20, 0.143, 0.2) == "caution"
    assert "Caution" in sentence("caution", 0.4, "") and sentence("can't judge", None, "no photo").endswith("no photo.")


def test_glucose_rules_run_before_the_model():
    """Full 5-minute readings pass; a 40-minute hole before the meal and an impossible value do not."""
    cfg = load_config()
    full = list(range(0, 120, 5))
    assert window_problem(_window(full), cfg) is None
    assert "incomplete" in window_problem(_window([m for m in full if m >= 40]), cfg)
    assert "outside" in window_problem(_window(full, 20.0), cfg)


def test_unfamiliar_photo_distance():
    """A copy of a training photo is at distance 0; an opposite direction is far; leave-one-out skips itself."""
    ref = torch.nn.functional.normalize(torch.randn(20, 8), dim=1)
    assert torch.isclose(knn_distance(ref, ref[:1], 1), torch.zeros(1), atol=1e-6).all()
    assert knn_distance(ref, -ref[:1], 1)[0] > 1e-3
    assert (leave_one_out_distances(ref, 1) > 1e-6).all()
