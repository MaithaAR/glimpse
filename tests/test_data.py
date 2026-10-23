"""Meal records are validated, and the new pipeline reproduces the proposal's meal counts exactly."""

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from glimpse.config import load_config
from glimpse.schemas import MealRecord

GOOD = dict(sid=3, t=datetime(2020, 5, 1, 12, 0), group="prediabetes", base=110.0, rise=62.0, coverage=0.9,
            share=100.0, spike50=1, carbs_served=55.0, photo="a.jpg", usable=True, onboarding=False,
            post_onboarding=True, next_meal_within=False)  # fmt: skip


def test_valid_meal_passes():
    """A plausible meal is accepted."""
    assert MealRecord.model_validate(GOOD).spike50 == 1


@pytest.mark.parametrize("field,value", [("group", "unknown"), ("spike50", 2), ("base", 5.0), ("coverage", 1.5)])
def test_malformed_meal_is_rejected(field, value):
    """Unknown group, a label outside {0,1}, an impossible glucose or coverage are rejected."""
    with pytest.raises(ValidationError):
        MealRecord.model_validate(GOOD | {field: value})


def test_label_must_match_rise():
    """A spike label that contradicts the measured rise is rejected."""
    with pytest.raises(ValidationError):
        MealRecord.model_validate(GOOD | {"rise": 20.0})


@pytest.mark.skipif(not Path("data/cgmacros/CGMacros/bio.csv").is_file(), reason="CGMacros not downloaded")
def test_reproduces_proposal_counts(tmp_path):
    """Same numbers as the proposal: 1,526 usable, 453 onboarding, 1,054 operational (769 target),
    896 isolated (634 target, 31 people), and no row rejected by the schema."""
    from glimpse.data.meals import build_meal_table, meal_counts

    cfg = load_config()
    M = build_meal_table(cfg, out_csv=tmp_path / "meals.csv")
    assert meal_counts(M, cfg) == {"usable": 1526, "onboarding": 453, "operational": 1054, "operational_target": 769,
                                   "isolated": 896, "isolated_target": 634, "isolated_target_people": 31}  # fmt: skip
