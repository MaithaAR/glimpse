"""The logged-carbs check uses the right inputs, never trains on a held-out person and applies the fixed rule."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from glimpse.data.dataset import PROFILE, MealDataset
from glimpse.schemas import MealSample
from glimpse.training.baselines import TREND_FEATURES
from glimpse.training.fusion import arm_features, carbs_check_decision, logged_carbs

HAVE_DATA = Path("data/dataset.json").is_file() and Path("splits/dev_splits.json").is_file()


def _toy(carbs: list[float | None]) -> MealDataset:
    n = len(carbs)
    samples = [MealSample(person_id=1, meal_time=datetime(2020, 5, 1, 12), group="healthy", photo_relpath=None,
                          baseline_mgdl=100.0, rise_mgdl=10.0, spike=0, carbs_served_g=c, onboarding=False,
                          isolated=True, premeal_ok=True) for c in carbs]  # fmt: skip
    return MealDataset(samples=samples, glucose=np.full((n, 24), 100.0), heart_rate=np.full((n, 24), 70.0),
                       profile=np.zeros((n, len(PROFILE))), profile_names=PROFILE)  # fmt: skip


def test_logged_carbs_column():
    """log(1 + grams), and NaN (filled later from training people) where nothing was logged."""
    v = logged_carbs(_toy([0.0, 50.0, None]), np.arange(3))
    assert v.shape == (3, 1) and v[0, 0] == 0.0 and np.isclose(v[1, 0], np.log(51)) and np.isnan(v[2, 0])


def test_arms_differ_only_by_the_carbs_column():
    """Both arms share the trend + profile inputs; the carbs arm adds exactly one column."""
    ds, rows = _toy([10.0, 20.0]), np.arange(2)
    a, b = arm_features("no_meal")(ds, rows), arm_features("logged_carbs")(ds, rows)
    assert a.shape[1] == len(TREND_FEATURES) + len(PROFILE) and b.shape[1] == a.shape[1] + 1
    assert np.array_equal(a, b[:, :-1], equal_nan=True)
    with pytest.raises(ValueError):
        arm_features("photo_magic")


def test_decision_rule():
    """Stop only when the whole 95% range is below the minimum gain; unclear results go ahead."""
    assert carbs_check_decision(0.015, 0.02).startswith("stop")
    assert carbs_check_decision(0.05, 0.02).startswith("proceed")
    assert carbs_check_decision(0.02, 0.02).startswith("proceed")


def test_grid_reaches_below_the_old_minimum():
    """The C grid was widened because the old minimum (0.01) kept winning."""
    from glimpse.config import load_config

    grid = load_config().baselines.logreg_c_grid
    assert min(grid) < 0.01 and grid == sorted(grid)


@pytest.mark.skipif(not HAVE_DATA, reason="run `just prepare` first")
def test_every_scored_meal_has_logged_carbs():
    """The check relies on logged carbs for every scored development target meal."""
    from glimpse.data.splits import Splits, meal_rows

    ds, sp = MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json"))
    rows = meal_rows(ds, {p for f in sp.outer_folds for p in f}, scored_only=True)
    assert len(rows) == 481 and all(ds.samples[i].carbs_served_g is not None for i in rows)


def test_photo_inputs_and_missing_photos():
    """Photo inputs = log grams + 3 bin probabilities; a meal without a photo gets NaN (filled from training)."""
    import pandas as pd

    from glimpse.training.fusion import PHOTO_COLS, photo_table, with_table

    df = pd.DataFrame({"row": [0], "photo_g": [50.0], "p_bin0": [0.1], "p_bin1": [0.2], "p_bin2": [0.3],
                       "p_bin3": [0.4]})  # fmt: skip
    t = photo_table(df)
    assert np.isclose(t[0][0], np.log(51)) and t[0][1:] == [0.2, 0.3, 0.4]
    x = with_table(t, len(PHOTO_COLS))(_toy([10.0, 20.0]), np.arange(2))
    assert x.shape[1] == len(TREND_FEATURES) + len(PROFILE) + len(PHOTO_COLS)
    assert not np.isnan(x[0, -4:]).any() and np.isnan(x[1, -4:]).all()


def test_shuffled_control_only_moves_photo_columns():
    """The control swaps the photo inputs between meals and leaves every other input where it was."""
    from glimpse.training.fusion import shuffle_within

    x = np.arange(40, dtype=float).reshape(10, 4)
    s = shuffle_within(x, 2, seed=1)
    assert np.array_equal(s[:, :2], x[:, :2])
    assert sorted(map(tuple, s[:, 2:])) == sorted(map(tuple, x[:, 2:])) and not np.array_equal(s, x)
