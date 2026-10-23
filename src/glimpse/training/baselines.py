"""Simple comparison models for the spike risk. They see the same meals and the same 2-h windows as the CNN, but
as a handful of hand-made trend numbers instead of the raw 24 slots."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glimpse.data.dataset import MealDataset

TREND_FEATURES = ["glucose_last", "slope_30min", "slope_60min", "glucose_mean_2h", "glucose_sd_2h", "change_2h",
                  "hr_mean_2h"]  # fmt: skip


def _slope(v: np.ndarray, n_slots: int) -> float:
    """Straight-line slope (mg/dL per minute) over the last n 5-minute slots; NaN if fewer than 3 readings."""
    last = v[-n_slots:]
    ok = ~np.isnan(last)
    return float(np.polyfit(np.arange(n_slots)[ok] * 5.0, last[ok], 1)[0]) if ok.sum() >= 3 else float("nan")


def trend_features(ds: MealDataset, rows: np.ndarray) -> np.ndarray:
    """Hand-made trend numbers from the pre-meal window (+ the 8 profile inputs). NaN where not computable."""
    out = []
    for i in rows:
        g, h = ds.glucose[i], ds.heart_rate[i]
        seen = g[~np.isnan(g)]
        hr = float(np.nanmean(h)) if (~np.isnan(h)).any() else float("nan")
        out.append([seen[-1], _slope(g, 6), _slope(g, 12), seen.mean(), seen.std(), seen[-1] - seen[0], hr,
                    *ds.profile[i]])  # fmt: skip
    return np.array(out, dtype=float)


def profile_features(ds: MealDataset, rows: np.ndarray) -> np.ndarray:
    """Only the 8 profile inputs (who the person is, meal hour, personal prior) - no sensor window."""
    return ds.profile[rows].astype(float)


def logistic(c: float, seed: int):
    """Missing values filled with training medians, inputs scaled, then L2 logistic regression with strength C."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=c, max_iter=2000, random_state=seed))  # fmt: skip


def gbm(depth: int, seed: int):
    """Gradient-boosted shallow trees (the proposal's pilot baseline); handles missing values itself."""
    return HistGradientBoostingClassifier(max_depth=depth, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                          random_state=seed)  # fmt: skip
