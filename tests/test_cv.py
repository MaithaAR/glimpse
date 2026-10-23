"""Cross-validation never predicts a person with a model trained on them; trend features are computed correctly."""

from pathlib import Path

import numpy as np
import pytest

from glimpse.training.baselines import TREND_FEATURES, _slope

HAVE_DATA = Path("data/dataset.json").is_file() and Path("splits/dev_splits.json").is_file()


def test_slope_of_a_straight_line():
    """A rise of 1 mg/dL per 5-minute slot is a slope of 0.2 mg/dL per minute."""
    v = np.arange(24, dtype=float) + 100
    assert np.isclose(_slope(v, 6), 0.2) and np.isclose(_slope(v, 12), 0.2)
    assert np.isnan(_slope(np.full(24, np.nan), 6))
    assert len(TREND_FEATURES) == 7


@pytest.mark.skipif(not HAVE_DATA, reason="run `just prepare` first")
def test_outer_folds_are_clean_and_cover_everyone():
    """Each development target person is held out exactly once and is never among that fold's training people."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.training.cv import outer_training_people

    ds, sp = MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json"))
    groups = {s.person_id: s.group for s in ds.samples}
    held = []
    for k, fold in enumerate(sp.outer_folds):
        train = outer_training_people(sp, k, groups)
        assert not set(train) & set(fold) and not set(train) & set(sp.locked)
        held += fold
    target_dev = [p for p, g in groups.items() if g != "T2D" and p not in sp.locked]
    assert sorted(held) == sorted(target_dev)
