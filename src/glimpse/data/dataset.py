"""Turn the meal table into model-ready examples: for every usable meal, the 2-h glucose and heart-rate windows
before it (via preprocess.py, the same code the demo uses), 8 profile inputs, the label and quality flags."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from glimpse.config import GlimpseConfig
from glimpse.data.preprocess import passes_premeal_checks, resample_window
from glimpse.schemas import MealSample

PROFILE = ["age", "bmi", "a1c_pct", "fasting_glucose_mgdl", "hour_sin", "hour_cos", "prior_spike_rate", "prior_rise"]


class MealDataset(BaseModel):
    """Validated samples plus aligned arrays: row i of every array belongs to samples[i]."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    samples: list[MealSample]
    glucose: np.ndarray  # (n, 24) mg/dL, NaN where a slot had no reading
    heart_rate: np.ndarray  # (n, 24) beats per minute, NaN where missing
    profile: np.ndarray  # (n, 8) in PROFILE order, NaN where unknown
    profile_names: list[str]

    def labels(self) -> np.ndarray:
        """Spike labels (0/1) in row order."""
        return np.array([s.spike for s in self.samples])

    def save(self, path: Path) -> None:
        """Arrays to <path>.npz, samples to <path>.json (human-readable)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), glucose=self.glucose, heart_rate=self.heart_rate,
                            profile=self.profile)  # fmt: skip
        meta = {"profile_names": self.profile_names, "samples": [s.model_dump(mode="json") for s in self.samples]}
        path.with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> MealDataset:
        """Read back and re-validate every sample."""
        arr = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text())
        samples = [MealSample.model_validate_json(json.dumps(s)) for s in meta["samples"]]
        return cls(samples=samples, glucose=arr["glucose"], heart_rate=arr["heart_rate"], profile=arr["profile"],
                   profile_names=meta["profile_names"])  # fmt: skip


def _person_series(root: Path, sid: int) -> pd.DataFrame:
    """One person's Dexcom glucose and Fitbit heart rate, indexed by time."""
    d = pd.read_csv(root / f"CGMacros-{sid:03d}" / f"CGMacros-{sid:03d}.csv")
    d.columns = [c.strip() for c in d.columns]
    hr = pd.to_numeric(d["HR"], errors="coerce") if "HR" in d else np.nan
    out = pd.DataFrame({"glucose": pd.to_numeric(d["Dexcom GL"], errors="coerce"), "hr": hr})
    return out.set_index(pd.to_datetime(d["Timestamp"])).sort_index()


def personal_prior(M: pd.DataFrame, min_meals: int) -> pd.DataFrame:
    """Each person's own spike rate and mean rise over their onboarding meals; NaN if they have too few.
    Missing values are filled later with the TRAINING people's average only (no information from other splits)."""
    on = M[M.onboarding & M.usable]
    st = on.groupby("sid").agg(n=("spike50", "size"), rate=("spike50", "mean"), rise=("rise", "mean"))
    st.loc[st.n < min_meals, ["rate", "rise"]] = np.nan
    return st[["rate", "rise"]]


def build_dataset(cfg: GlimpseConfig, meals_csv: Path = Path("data/meals.csv")) -> MealDataset:
    """Every usable, labelled meal -> validated sample + glucose/heart-rate windows + profile inputs."""
    log = logger.bind(stage="data")
    M = pd.read_csv(meals_csv, parse_dates=["t"])
    M = M[M.usable & M.rise.notna()].reset_index(drop=True)
    prior = personal_prior(M, cfg.onboarding.min_meals_for_personal_prior)
    root, pc = Path(cfg.data.cgmacros_root), cfg.pipeline
    samples, glu, hr, prof, rejected = [], [], [], [], 0
    series: dict[int, pd.DataFrame] = {}
    for r in M.itertuples(index=False):
        s = series.setdefault(r.sid, _person_series(root, r.sid))
        win = s.loc[r.t - pd.Timedelta(minutes=pc.cgm_lookback_min) : r.t]  # never after the meal
        times = list(win.index.to_pydatetime())
        g = resample_window(times, list(win.glucose), r.t.to_pydatetime(), pc.cgm_lookback_min)
        h = resample_window(times, list(win.hr), r.t.to_pydatetime(), pc.cgm_lookback_min)
        ok = passes_premeal_checks(g, pc.min_cgm_readings_last_30min, pc.min_cgm_coverage, pc.max_cgm_gap_min)
        carbs = None if pd.isna(r.carbs_served) else float(r.carbs_served)
        try:
            sample = MealSample(
                person_id=int(r.sid), meal_time=r.t.to_pydatetime(), group=str(r.group),
                photo_relpath=f"CGMacros-{r.sid:03d}/{r.photo.strip()}" if isinstance(r.photo, str) else None,
                baseline_mgdl=float(r.base), rise_mgdl=float(r.rise), spike=int(r.spike50), carbs_served_g=carbs,
                onboarding=bool(r.onboarding), isolated=bool(r.post_onboarding and not r.next_meal_within),
                premeal_ok=bool(ok),
            )  # fmt: skip
        except ValidationError as e:
            rejected += 1
            log.warning("rejected meal sid={} t={}: {}", r.sid, r.t, e.errors()[0]["msg"])
            continue
        hour = r.t.hour + r.t.minute / 60
        pr = prior.loc[r.sid] if r.sid in prior.index else pd.Series({"rate": np.nan, "rise": np.nan})
        samples.append(sample)
        glu.append(g.values)
        hr.append(h.values)
        prof.append([r.Age, r.BMI, r.a1c, r.fglu, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
                     pr["rate"], pr["rise"]])  # fmt: skip
    n_slots = pc.cgm_lookback_min // 5
    ds = MealDataset(samples=samples, glucose=np.array(glu), heart_rate=np.array(hr),
                     profile=np.array(prof, dtype=float), profile_names=PROFILE)  # fmt: skip
    assert ds.glucose.shape == ds.heart_rate.shape == (len(samples), n_slots), "window shape mismatch"
    assert ds.profile.shape == (len(samples), len(PROFILE)), "profile shape mismatch"
    log.info("dataset: {} meals ({} isolated), {} rejected by schema, {} fail the pre-meal checks; windows {}",
             len(samples), sum(s.isolated for s in samples), rejected, sum(not s.premeal_ok for s in samples),
             ds.glucose.shape)  # fmt: skip
    return ds
