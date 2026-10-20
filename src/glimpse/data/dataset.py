"""Labelled meal dataset: validated samples plus pre-meal sensor windows and profile features as arrays."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from glimpse.config import GlimpseConfig
from glimpse.data.meals import add_personal_prior, build_meals
from glimpse.data.preprocess import resample_window
from glimpse.schemas import MealSample

TABULAR = ["age", "bmi", "a1c_pct", "fasting_glucose_mgdl", "hour_sin", "hour_cos", "prior_spike_rate", "prior_rise"]


class MealDataset(BaseModel):
    """All usable meals: validated metadata and aligned arrays (row i of every array is samples[i])."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    samples: list[MealSample]
    glucose: np.ndarray  # (n, 24) mg/dL on the 5-min grid, NaN where missing
    heart_rate: np.ndarray  # (n, 24) bpm, NaN where missing
    tabular: np.ndarray  # (n, len(TABULAR)), NaN where unknown
    tabular_names: list[str]

    def person_ids(self) -> np.ndarray:
        """Person id per row."""
        return np.array([s.person_id for s in self.samples])

    def labels(self) -> np.ndarray:
        """Spike label per row."""
        return np.array([s.spike for s in self.samples])

    def save(self, path: Path) -> None:
        """Write arrays to .npz and validated samples to a JSON sidecar."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, glucose=self.glucose, heart_rate=self.heart_rate, tabular=self.tabular)
        path.with_suffix(".json").write_text(
            json.dumps(
                {"tabular_names": self.tabular_names, "samples": [s.model_dump(mode="json") for s in self.samples]}
            )
        )

    @classmethod
    def load(cls, path: Path) -> MealDataset:
        """Read a dataset written by `save`, re-validating every sample."""
        arr = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        samples = [MealSample.model_validate(s) for s in meta["samples"]]
        return cls(samples=samples, glucose=arr["glucose"], heart_rate=arr["heart_rate"], tabular=arr["tabular"],
                   tabular_names=meta["tabular_names"])  # fmt: skip


def _person_series(root: Path, sid: int) -> pd.DataFrame:
    """Dexcom glucose and Fitbit heart rate for one person, indexed by timestamp."""
    f = root / f"CGMacros-{sid:03d}" / f"CGMacros-{sid:03d}.csv"
    d = pd.read_csv(f)
    d.columns = [c.strip() for c in d.columns]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    out = pd.DataFrame(
        {"glucose": pd.to_numeric(d["Dexcom GL"], errors="coerce"), "hr": pd.to_numeric(d.get("HR"), errors="coerce")}
    )
    return out.set_index(d["Timestamp"]).sort_index()


def _personal_prior(meals: pd.DataFrame, cfg: GlimpseConfig) -> pd.DataFrame:
    """Spike rate and mean rise from each person's own onboarding meals; NaN if too few (imputed from train later)."""
    on = meals[meals.onboarding & meals.usable]
    st = on.groupby("sid").agg(n=("spike50", "size"), rate=("spike50", "mean"), rise=("rise", "mean"))
    few = st.n < cfg.onboarding.min_meals_for_personal_prior
    st.loc[few, ["rate", "rise"]] = np.nan
    return st[["rate", "rise"]]


def build_dataset(cfg: GlimpseConfig) -> MealDataset:
    """Build the dataset from raw CGMacros files using the same window code the product uses at inference."""
    log = logger.bind(stage="data")
    meals = add_personal_prior(build_meals(cfg), cfg)
    meals = meals[meals.usable & meals.rise.notna()].reset_index(drop=True)
    prior = _personal_prior(meals, cfg)
    root, lb, pc = Path(cfg.data.cgmacros_root), cfg.pipeline.cgm_lookback_min, cfg.pipeline
    samples, glu, hr, tab = [], [], [], []
    rejected = 0
    series: dict[int, pd.DataFrame] = {}
    for r in meals.itertuples(index=False):
        s = series.setdefault(r.sid, _person_series(root, r.sid))
        win = s.loc[r.t - pd.Timedelta(minutes=lb) : r.t]
        g = resample_window(list(win.index.to_pydatetime()), list(win.glucose), r.t.to_pydatetime(), lb)
        h = resample_window(list(win.index.to_pydatetime()), list(win.hr), r.t.to_pydatetime(), lb)
        premeal_ok = (
            g.readings_last_30min >= pc.min_cgm_readings_last_30min
            and g.coverage >= pc.min_cgm_coverage
            and g.max_gap_min <= pc.max_cgm_gap_min
        )
        try:
            sample = MealSample(
                person_id=int(r.sid),
                meal_time=r.t.to_pydatetime(),
                group=r.group,
                photo_relpath=f"CGMacros-{r.sid:03d}/{r.photo.strip()}" if isinstance(r.photo, str) else None,
                baseline_mgdl=float(r.base),
                rise_mgdl=float(r.rise),
                spike=int(r.spike50),
                carbs_served_g=None if pd.isna(r.carbs_served) else float(r.carbs_served),
                onboarding=bool(r.onboarding),
                isolated=bool(r.post_onboarding and not r.next_meal_within),
                premeal_ok=bool(premeal_ok),
            )
        except ValidationError as e:
            rejected += 1
            log.warning("rejected meal person={} t={}: {}", r.sid, r.t, str(e).splitlines()[1:3])
            continue
        hour = r.t.hour + r.t.minute / 60
        pr = prior.loc[r.sid] if r.sid in prior.index else pd.Series({"rate": np.nan, "rise": np.nan})
        samples.append(sample)
        glu.append(g.values)
        hr.append(h.values)
        tab.append([r.Age, r.BMI, r.a1c, r.fglu, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
                    pr["rate"], pr["rise"]])  # fmt: skip
    ds = MealDataset(samples=samples, glucose=np.array(glu), heart_rate=np.array(hr),
                     tabular=np.array(tab, dtype=float), tabular_names=TABULAR)  # fmt: skip
    assert ds.glucose.shape == ds.heart_rate.shape == (len(samples), lb // 5), "sensor window shape mismatch"
    assert ds.tabular.shape == (len(samples), len(TABULAR)), "tabular shape mismatch"
    log.info("dataset: {} meals kept, {} rejected by schema, {} fail pre-meal checks",
             len(samples), rejected, sum(not s.premeal_ok for s in samples))  # fmt: skip
    return ds
