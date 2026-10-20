"""Data-stage tests on synthetic data (no CGMacros files needed)."""

from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from glimpse.config import load_config
from glimpse.data.augment import augment_sequence
from glimpse.data.dataset import TABULAR, MealDataset
from glimpse.data.normalise import Normaliser
from glimpse.data.quality import dhash
from glimpse.data.splits import Splits, make_splits
from glimpse.schemas import MealSample

CFG = load_config("configs/default.yaml")


def _dataset(n_people: int = 45, meals_each: int = 4) -> MealDataset:
    """Synthetic dataset: groups cycle healthy / prediabetes / T2D so the target cohort is large enough."""
    rng = np.random.default_rng(0)
    groups = ["healthy", "prediabetes", "T2D"]
    samples = [
        MealSample(person_id=p, meal_time=datetime(2026, 1, 1) + timedelta(hours=6 * k), group=groups[p % 3],
                   photo_relpath=None, baseline_mgdl=100.0, rise_mgdl=float(rng.normal(40, 20)), spike=int(k % 2),
                   onboarding=k == 0, isolated=k > 0, premeal_ok=True)
        for p in range(1, n_people + 1) for k in range(meals_each)
    ]  # fmt: skip
    n = len(samples)
    glu = rng.normal(110, 20, (n, 24))
    glu[0, 3:6] = np.nan
    return MealDataset(samples=samples, glucose=glu, heart_rate=rng.normal(75, 10, (n, 24)),
                       tabular=rng.normal(0, 1, (n, len(TABULAR))), tabular_names=TABULAR)  # fmt: skip


def test_meal_sample_rejects_bad_values():
    """Schema validation rejects impossible labels and rises."""
    ok = {"person_id": 1, "meal_time": datetime(2026, 1, 1), "group": "healthy", "photo_relpath": None,
          "baseline_mgdl": 100.0, "rise_mgdl": 30.0, "spike": 0, "onboarding": False, "isolated": True,
          "premeal_ok": True}  # fmt: skip
    MealSample(**ok)
    for bad in ({"spike": 2}, {"rise_mgdl": 900.0}, {"group": "unknown"}):
        with pytest.raises(ValidationError):
            MealSample(**(ok | bad))


def test_splits_person_disjoint_locked_excluded_and_seeded(tmp_path):
    """Splits never share a person, never include locked people, and are reproducible from the seed."""
    ds = _dataset()
    a, b = make_splits(ds, CFG), make_splits(ds, CFG)
    assert a == b
    assert not set(a.locked) & (set(a.train) | set(a.validation) | set(a.test))
    assert all(ds.samples[[s.person_id for s in ds.samples].index(p)].group != "T2D" for p in a.test + a.validation)
    a.save(tmp_path / "s.json")
    assert Splits.load(tmp_path / "s.json") == a
    with pytest.raises(ValidationError):
        Splits(seed=0, train=[1, 2], validation=[2], test=[3], cv_folds=[], locked=[4])


def test_normaliser_uses_training_statistics_only():
    """Statistics come from the rows passed to fit; transform fills gaps and returns finite, shaped arrays."""
    ds = _dataset()
    tr = slice(0, 60)
    norm = Normaliser.fit(ds.glucose[tr], ds.heart_rate[tr], ds.tabular[tr], ds.tabular_names)
    assert norm.glucose_mean == pytest.approx(np.nanmean(ds.glucose[tr]))
    x = norm.transform(ds.glucose, ds.heart_rate, ds.tabular)
    assert x.sequence.shape == (len(ds.samples), 3, 24) and np.isfinite(x.sequence).all()
    assert x.sequence[0, 2, 3:6].tolist() == [1.0, 1.0, 1.0]  # missing-glucose mask channel


def test_augmentation_keeps_shape_and_mask():
    """Augmentation preserves shape, never adds noise to the mask channel, and changes the signal."""
    seq = np.zeros((4, 3, 24), dtype=np.float32)
    seq[:, 2, 5] = 1.0
    out = augment_sequence(seq, np.random.default_rng(1))
    assert out.shape == seq.shape and set(np.unique(out[:, 2])) <= {0.0, 1.0}
    assert not np.allclose(out[:, :2], 0)


def test_dataset_roundtrip(tmp_path):
    """Save/load keeps arrays and re-validates samples."""
    ds = _dataset(6, 2)
    ds.save(tmp_path / "d.npz")
    back = MealDataset.load(tmp_path / "d.npz")
    assert back.samples == ds.samples and np.array_equal(back.glucose, ds.glucose, equal_nan=True)


def test_dhash_detects_duplicates(tmp_path):
    """Identical images hash identically; different images differ by many bits."""
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    a.save(tmp_path / "a.png")
    a.resize((60, 60)).save(tmp_path / "a_small.png")
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "b.png")
    assert (dhash(tmp_path / "a.png") ^ dhash(tmp_path / "a_small.png")).bit_count() <= 4
    assert (dhash(tmp_path / "a.png") ^ dhash(tmp_path / "b.png")).bit_count() > 10
