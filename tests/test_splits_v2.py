"""Partition v2 (design D1): the schema rejects any malformed partition, the builder is deterministic,
and the reconciliation proves meal-level once-only coverage -- the double-count class of error."""

from pathlib import Path

import pytest

from glimpse.config import load_config
from glimpse.data.splits import SplitsV2, assert_not_locked


def _ok() -> dict:
    return {"seed": 1, "locked": [90, 91], "dev_folds": [[1, 2], [3, 4]], "t2d": [50, 51],
            "inner_folds": [[[3], [4]], [[1], [2]]]}  # fmt: skip


def test_schema_accepts_a_proper_partition():
    sp = SplitsV2(**_ok())
    assert sp.training_people(0) == [3, 4, 50, 51] and sp.training_people(1) == [1, 2, 50, 51]


def test_schema_rejects_a_person_in_two_folds():
    bad = _ok() | {"dev_folds": [[1, 2], [2, 4]]}
    with pytest.raises(ValueError, match="more than one fold"):
        SplitsV2(**bad)


def test_schema_rejects_t2d_on_an_inner_validation_side():
    bad = _ok() | {"inner_folds": [[[3], [4, 50]], [[1], [2]]]}
    with pytest.raises(ValueError, match="inner folds"):
        SplitsV2(**bad)


def test_schema_rejects_locked_person_in_dev():
    bad = _ok() | {"dev_folds": [[1, 90], [3, 4]], "inner_folds": [[[3], [4]], [[1], [90]]]}
    with pytest.raises(ValueError, match="share people"):
        SplitsV2(**bad)


def test_assert_not_locked_is_a_hard_error():
    assert_not_locked([1, 2], [90, 91], "test") # fine
    with pytest.raises(RuntimeError, match=r"locked people \[90\]"):
        assert_not_locked([1, 90], [90, 91], "test")


needs_data = pytest.mark.skipif(not Path("data/dataset.json").is_file(), reason="dataset not built")


@needs_data
def test_make_splits_v2_is_deterministic_and_well_formed():
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import make_splits_v2

    cfg = load_config()
    ds = MealDataset.load(Path("data/dataset"))
    a, b = make_splits_v2(ds, cfg), make_splits_v2(ds, cfg)
    assert a == b  # same seed, same dataset -> same partition
    assert len(a.dev_folds) == cfg.cv.outer_folds and len(a.locked) == 8
    assert all(len(inner) == cfg.cv.inner_folds for inner in a.inner_folds)


@needs_data
def test_reconciliation_asserts_and_reports():
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits, make_splits_v2, reconciliation_v2

    cfg = load_config()
    ds = MealDataset.load(Path("data/dataset"))
    md = reconciliation_v2(ds, Splits.load(Path("splits/dev_splits.json")), make_splits_v2(ds, cfg))
    assert "exactly once" in md and "| fold | people |" in md.replace("held-out meals |", "held-out meals |")
    assert "Where the intermediate's fixed groups landed" in md
