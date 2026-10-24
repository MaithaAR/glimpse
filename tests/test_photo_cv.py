"""Photo cross-validation: the recipe rule, no person ever predicted by a model that saw them, and the baselines."""

from pathlib import Path

import numpy as np
import pytest

from glimpse.config import load_config
from glimpse.training.photo import choose_recipe
from glimpse.training.photo_cv import _one_hot_bins, _ridge, _ridge_predict, fold_plan

HAVE_DATA = Path("data/dataset.json").is_file() and Path("splits/dev_splits.json").is_file()


def test_recipe_rule():
    """Median head epoch; last stage kept only with a big enough gain, for its best epoch."""
    hp = load_config().training.photo
    assert choose_recipe([10, 20, 30], [30.0] * 3, [], hp) == (20, False, 0)
    assert choose_recipe([15, 15], [30.0, 30.0], [[28.0, 25.0, 27.0], [29.0, 26.0, 28.0]], hp) == (15, True, 2)
    assert choose_recipe([15, 15], [30.0, 30.0], [[29.9, 30.1], [29.9, 30.2]], hp) == (15, False, 0)


def test_baseline_helpers():
    """Ridge estimates are never negative; a gram value on a bin edge goes to the upper bin."""
    rng = np.random.default_rng(0)
    emb, g = rng.normal(size=(50, 8)), rng.uniform(0, 120, 50)
    assert (_ridge_predict(_ridge(emb, g, 10.0), emb) >= 0).all()
    assert _one_hot_bins(np.array([10.0, 30.0, 60.0, 95.0])).argmax(1).tolist() == [0, 1, 2, 3]


@pytest.mark.skipif(not HAVE_DATA, reason="run `just prepare` first")
@pytest.mark.parametrize("k", range(5))
def test_no_person_is_predicted_by_a_model_that_saw_them(k):
    """Outer fold k: held-out and locked people never train; the inner folds split the training people exactly
    once, so every training person gets one nested estimate from a model without them."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits

    ds, sp = MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json"))
    groups = {s.person_id: s.group for s in ds.samples}
    f = fold_plan(sp, groups, load_config())[k]
    train = set(f["train"])
    assert not train & set(f["held"]) and not train & set(sp.locked) and not set(f["held"]) & set(sp.locked)
    inner_people = [p for held in f["inner"] for p in held]
    assert sorted(inner_people) == sorted(train)  # each training person held out exactly once
