"""v2, the sensor-first product (docs/v2_sensor_first.md): the risk never depends on the photo, a missing or
unusable photo withholds only the carb assist, v1's frozen files stay untouched, and the v2 freeze runs once."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tests.test_serve import MEAL, PROFILE, bare_pipeline, flat_cgm

from glimpse.serve import PhotoEstimate
from glimpse.serve.schemas import MealInput

PRIORS = {**PROFILE, "prior_spike_rate": 0.3, "prior_rise": 40.0}


class _Recorder:
    """Stub fusion that records how many inputs it was given (v2 must pass the 15 trend + profile only)."""

    def __init__(self, risk: float):
        self.risk, self.widths = risk, []

    def predict_proba(self, x):
        self.widths.append(np.asarray(x).shape[1])
        return np.array([[1 - self.risk, self.risk]])


def _v2(risk: float, table: dict) -> object:
    p = bare_pipeline(risk=risk, table=table)
    p.sensor_first, p.fusion = True, _Recorder(risk)
    return p


def test_v2_answers_without_a_photo_and_flags_the_missing_carb_assist():
    p = _v2(0.45, {})
    a = p.answer(MealInput(meal_time=MEAL, profile=PRIORS, **flat_cgm()), key=0)
    assert a.decision == "caution" and a.risk == pytest.approx(0.45)
    assert "carbs_unavailable_photo_present" in a.flags and a.carbs_g is None and a.carb_source == "none"


def test_v2_risk_does_not_depend_on_the_photo():
    small = PhotoEstimate(grams=10.0, bin_probs=[1.0, 0.0, 0.0, 0.0], unfamiliar_score=0.2)
    big = PhotoEstimate(grams=150.0, bin_probs=[0.0, 0.0, 0.0, 1.0], unfamiliar_score=0.2)
    m = MealInput(meal_time=MEAL, profile=PRIORS, **flat_cgm())
    p = _v2(0.3, {0: small, 1: big})
    a, b = p.answer(m, key=0), p.answer(m, key=1)
    assert a.risk == b.risk and a.carbs_g == 10.0 and b.carbs_g == 150.0 and a.carb_source == "photo"
    assert set(p.fusion.widths) == {15}  # 7 trend + 8 profile inputs, no photo columns


def test_v2_keeps_r13_and_the_typed_carbs_are_shown_not_used():
    p = _v2(0.05, {})
    bare = p.answer(MealInput(meal_time=MEAL, profile=PROFILE, **flat_cgm()), key=0)
    assert bare.decision == "can't judge" and "rule R13" in bare.reason
    typed = p.answer(MealInput(meal_time=MEAL, profile=PRIORS, user_carbs_g=80.0, **flat_cgm()), key=0)
    assert typed.decision == "low risk" and "carbs_from_user_shown_not_used" in typed.flags


def test_v1_frozen_files_are_untouched_and_v2_files_are_separate():
    v1 = json.loads(Path("checkpoints/thresholds.json").read_text())
    assert v1["sha256"].startswith("34607e06e247") and v1["main"] == {"low_risk_below": 0.15, "caution_from": 0.16}
    v2 = json.loads(Path("checkpoints/sensor_first/thresholds.json").read_text())
    from glimpse.training.final import hash_thresholds

    assert hash_thresholds(v2) == v2["sha256"] and "v2 sensor-first" in v2["provenance"]


def test_v2_freeze_refuses_a_second_run():
    from glimpse.config import load_config
    from glimpse.training.finalize import finalize_sensor_first

    with pytest.raises(RuntimeError, match="written once"):
        finalize_sensor_first(None, None, load_config())
