"""Spike-risk baselines fitted on the training people only: prevalence, personal prior, logistic regression and GBM
on engineered sensor features (the same raw window and profile inputs the sensor CNN sees)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glimpse.data.dataset import MealDataset

SENSOR_FEATURES = ["glucose_last", "glucose_mean_2h", "glucose_sd_2h", "slope_30min", "slope_60min",
                   "change_2h", "hr_mean_2h"]  # fmt: skip
PRIOR_RATE = "prior_spike_rate"


def _slope(v: np.ndarray, n: int) -> float:
    """Least-squares slope (mg/dL per min) over the last `n` 5-min readings; NaN if fewer than 3 are present."""
    last = v[-n:]
    ok = ~np.isnan(last)
    return float(np.polyfit(np.arange(n)[ok] * 5.0, last[ok], 1)[0]) if ok.sum() >= 3 else float("nan")


def engineered_features(ds: MealDataset, rows: np.ndarray) -> np.ndarray:
    """Hand-made trend features from the 2-h window plus the 8 profile/time/prior inputs (NaN where unknown)."""
    out = []
    for i in rows:
        g, h = ds.glucose[i], ds.heart_rate[i]
        present = g[~np.isnan(g)]
        out.append([present[-1], present.mean(), present.std(), _slope(g, 6), _slope(g, 12),
                    present[-1] - present[0], np.nanmean(h) if (~np.isnan(h)).any() else np.nan,
                    *ds.tabular[i]])  # fmt: skip
    return np.array(out, dtype=float)


def feature_names(ds: MealDataset) -> list[str]:
    """Column names of `engineered_features`."""
    return SENSOR_FEATURES + list(ds.tabular_names)


def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int):
    """Median imputation (training values) + scaling + L2 logistic regression (C = 1, not tuned)."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, random_state=seed)).fit(x, y)  # fmt: skip


def fit_gbm(x: np.ndarray, y: np.ndarray, seed: int):
    """Shallow gradient-boosted trees (fixed, untuned settings; missing values handled natively)."""
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                          random_state=seed).fit(x, y)  # fmt: skip


def personal_prior(ds: MealDataset, rows: np.ndarray, fallback: float) -> np.ndarray:
    """Each person's onboarding spike rate as the risk ('you usually spike'); training-cohort rate when unknown."""
    col = ds.tabular_names.index(PRIOR_RATE)
    v = ds.tabular[rows, col].astype(float)
    return np.where(np.isnan(v), fallback, v)
