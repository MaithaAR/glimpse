"""Splits keep people apart; the normaliser only learns from training people."""

from pathlib import Path

import numpy as np
import pytest

from glimpse.config import load_config
from glimpse.data.normalise import Normaliser

HAVE_DATA = Path("data/dataset.json").is_file()


@pytest.mark.skipif(not HAVE_DATA, reason="run `just prepare` first")
def test_splits_keep_people_apart_and_are_repeatable():
    """No person in two groups; locked people unused; validation/test are target people; same seed, same split."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import make_splits

    cfg = load_config()
    ds = MealDataset.load(Path("data/dataset"))
    a, b = make_splits(ds, cfg), make_splits(ds, cfg)
    assert a == b
    groups = {s.person_id: s.group for s in ds.samples}
    used = set(a.train) | set(a.validation) | set(a.test)
    assert not used & set(a.locked) and len(a.locked) == 8
    assert all(groups[p] != "T2D" for p in a.validation + a.test)
    assert a.test == a.outer_folds[0] and len(a.test) == 5 and len(a.validation) == 4
    assert sorted(sum(a.outer_folds, [])) == sorted(p for p in groups if p not in a.locked and groups[p] != "T2D")


def test_normaliser_ignores_other_splits():
    """Statistics come only from the rows passed in; the output has 3 channels x 24 slots and no gaps."""
    rng = np.random.default_rng(0)
    g, h, p = rng.normal(110, 15, (50, 24)), rng.normal(75, 8, (50, 24)), rng.normal(0, 1, (50, 8))
    g[0, 3] = np.nan
    p[1, 6] = np.nan
    norm = Normaliser.fit(g[:40], h[:40], p[:40], [f"x{i}" for i in range(8)])
    g_changed = g.copy()
    g_changed[40:] += 500  # change the "test" rows only
    assert Normaliser.fit(g_changed[:40], h[:40], p[:40], norm.profile_names) == norm
    x = norm.transform(g, h, p)
    assert x.sequence.shape == (50, 3, 24) and x.profile.shape == (50, 8)
    assert not np.isnan(x.sequence).any() and not np.isnan(x.profile).any()
    assert x.sequence[0, 2, 3] == 1.0  # the missing glucose slot is flagged


@pytest.mark.skipif(not HAVE_DATA, reason="run `just prepare` first")
def test_test_and_validation_people_never_used_for_fitting():
    """Inner (tuning) folds and the normaliser's rows contain training people only."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits, meal_rows

    ds = MealDataset.load(Path("data/dataset"))
    sp = Splits.load(Path("splits/dev_splits.json"))
    held_out = set(sp.test) | set(sp.validation) | set(sp.locked)
    assert not set(sum(sp.inner_folds, [])) & held_out
    assert sorted(sum(sp.inner_folds, [])) == sorted(sp.train)
    rows = meal_rows(ds, set(sp.train), scored_only=False)
    assert not {ds.samples[i].person_id for i in rows} & held_out


def test_constant_input_does_not_break_scaling():
    """If an input never changes in training (spread 0), scaling must not divide by zero."""
    g, h, p = np.full((10, 24), 100.0), np.full((10, 24), 70.0), np.zeros((10, 8))
    x = Normaliser.fit(g, h, p, [f"x{i}" for i in range(8)]).transform(g, h, p)
    assert np.isfinite(x.sequence).all() and np.isfinite(x.profile).all()
