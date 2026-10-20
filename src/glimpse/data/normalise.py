"""Feature scaling fitted on training people only and reused unchanged at validation, test and inference time."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict

from glimpse.data.preprocess import fill_gaps


class ModelInputs(BaseModel):
    """Scaled model inputs: sequence channels (glucose z, heart-rate z, glucose-missing mask) and tabular z."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    sequence: np.ndarray  # (n, 3, 24) float32
    tabular: np.ndarray  # (n, k) float32


class Normaliser(BaseModel):
    """Means and standard deviations from the training split; missing values are imputed with the training mean."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    glucose_mean: float
    glucose_std: float
    hr_mean: float
    hr_std: float
    tabular_names: list[str]
    tabular_mean: list[float]
    tabular_std: list[float]
    n_train_meals: int

    @classmethod
    def fit(cls, glucose: np.ndarray, heart_rate: np.ndarray, tabular: np.ndarray, names: list[str]) -> Normaliser:
        """Fit on training rows only (caller passes the training subset)."""
        tstd = np.nanstd(tabular, axis=0)
        n = cls(
            glucose_mean=float(np.nanmean(glucose)), glucose_std=float(np.nanstd(glucose)),
            hr_mean=float(np.nanmean(heart_rate)), hr_std=float(np.nanstd(heart_rate)),
            tabular_names=names, tabular_mean=[float(v) for v in np.nanmean(tabular, axis=0)],
            tabular_std=[float(v) if v > 0 else 1.0 for v in tstd], n_train_meals=len(glucose),
        )  # fmt: skip
        logger.bind(stage="normalise").info("fitted on {} training meals", len(glucose))
        return n

    def transform(self, glucose: np.ndarray, heart_rate: np.ndarray, tabular: np.ndarray) -> ModelInputs:
        """Gap-fill each window, z-score with training statistics, add a missing-glucose mask channel."""
        g = np.stack([fill_gaps(row, self.glucose_mean) for row in np.atleast_2d(glucose)])
        h = np.stack([fill_gaps(row, self.hr_mean) for row in np.atleast_2d(heart_rate)])
        mask = np.isnan(np.atleast_2d(glucose)).astype(float)
        seq = np.stack([(g - self.glucose_mean) / self.glucose_std, (h - self.hr_mean) / self.hr_std, mask], axis=1)
        t = np.atleast_2d(tabular).astype(float)
        t = np.where(np.isnan(t), np.array(self.tabular_mean), t)  # unknown -> training mean (population prior)
        t = (t - np.array(self.tabular_mean)) / np.array(self.tabular_std)
        return ModelInputs(sequence=seq.astype(np.float32), tabular=t.astype(np.float32))

    def save(self, path: Path) -> None:
        """Write to JSON (shipped with the model checkpoint)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Normaliser:
        """Read and validate."""
        return cls.model_validate(json.loads(path.read_text()))
