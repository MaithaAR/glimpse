"""Training learns on a toy problem, logs every epoch, and a saved checkpoint gives the same predictions."""

from datetime import datetime

import numpy as np

from glimpse.config import load_config
from glimpse.data.dataset import PROFILE, MealDataset
from glimpse.data.normalise import Normaliser
from glimpse.schemas import MealSample
from glimpse.training.common import EpochLogger
from glimpse.training.scratch import fit, load_bundle, predict_proba, save_bundle


def _toy(n: int = 200) -> MealDataset:
    """Meals whose label is 'glucose was rising before the meal' - an easy pattern a working loop must learn."""
    rng = np.random.default_rng(0)
    rising = rng.integers(0, 2, n)
    slope = np.where(rising == 1, 1.5, -0.5)[:, None] * np.arange(24)
    glucose = 100 + slope + rng.normal(0, 2, (n, 24))
    samples = [MealSample(person_id=1 + i % 10, meal_time=datetime(2020, 5, 1, 12), group="healthy",
                          photo_relpath=None, baseline_mgdl=100.0, rise_mgdl=60.0 if r else 10.0, spike=int(r),
                          carbs_served_g=None, onboarding=False, isolated=True, premeal_ok=True)
               for i, r in enumerate(rising)]  # fmt: skip
    return MealDataset(samples=samples, glucose=glucose, heart_rate=np.full((n, 24), 70.0),
                       profile=rng.normal(0, 1, (n, len(PROFILE))), profile_names=PROFILE)  # fmt: skip


def test_loss_goes_down_and_epochs_are_logged(tmp_path):
    """On an easy toy problem the training loss falls, and every epoch is written to the CSV file."""
    cfg = load_config()
    ds = _toy()
    rows = np.arange(len(ds.samples))
    norm = Normaliser.fit(ds.glucose, ds.heart_rate, ds.profile, PROFILE)
    elog = EpochLogger(tmp_path / "history.csv")
    fit(ds, rows, norm, cfg.training.scratch, cfg, epochs=30, seed=0, run="toy", elog=elog)
    elog.close()
    losses = [r.train_loss for r in elog.records]
    assert len(losses) == 30 and losses[-1] < 0.6 * losses[0]
    assert len((tmp_path / "history.csv").read_text().splitlines()) == 31  # header + 30 epochs


def test_checkpoint_round_trip(tmp_path):
    """Saving and reloading the bundle reproduces the same probabilities."""
    cfg = load_config()
    ds = _toy(60)
    rows = np.arange(len(ds.samples))
    norm = Normaliser.fit(ds.glucose, ds.heart_rate, ds.profile, PROFILE)
    model, _ = fit(ds, rows, norm, cfg.training.scratch, cfg, epochs=2, seed=0, run="toy")
    x = norm.transform(ds.glucose, ds.heart_rate, ds.profile)
    save_bundle(model, norm, cfg.training.scratch, tmp_path / "m.pt", epochs=2)
    model2, norm2, info = load_bundle(tmp_path / "m.pt")
    assert np.allclose(predict_proba(model, x), predict_proba(model2, norm2.transform(ds.glucose, ds.heart_rate,
                                                                                      ds.profile)))  # fmt: skip
    assert info["epochs"] == 2
