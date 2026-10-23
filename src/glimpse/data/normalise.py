"""Scaling fitted on TRAINING people only and reused unchanged for validation, test, cross-validation folds and
the demo. Each input becomes (value - training mean) / training spread, so all inputs share a similar range."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict

from glimpse.data.preprocess import fill_gaps


class ModelInputs(BaseModel):
    """What the sensor model receives: 3 channels x 24 slots (glucose, heart rate, missing-glucose flag) + profile."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    sequence: np.ndarray  # (n, 3, 24) float32
    profile: np.ndarray  # (n, 8) float32


class Normaliser(BaseModel):
    """Training-set averages and spreads; unknown values are replaced by the training average."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    glucose_mean: float
    glucose_std: float
    hr_mean: float
    hr_std: float
    profile_names: list[str]
    profile_mean: list[float]
    profile_std: list[float]
    n_train_meals: int

    @classmethod
    def fit(cls, glucose: np.ndarray, heart_rate: np.ndarray, profile: np.ndarray, names: list[str]) -> Normaliser:
        """Compute the statistics from the rows given (the caller passes training rows only)."""
        pstd = np.nanstd(profile, axis=0)
        n = cls(glucose_mean=float(np.nanmean(glucose)), glucose_std=float(np.nanstd(glucose)),
                hr_mean=float(np.nanmean(heart_rate)), hr_std=float(np.nanstd(heart_rate)), profile_names=list(names),
                profile_mean=[float(v) for v in np.nanmean(profile, axis=0)],
                profile_std=[float(v) if v > 0 else 1.0 for v in pstd], n_train_meals=len(glucose))  # fmt: skip
        logger.bind(stage="normalise").info("fitted on {} training meals", len(glucose))
        return n

    def transform(self, glucose: np.ndarray, heart_rate: np.ndarray, profile: np.ndarray) -> ModelInputs:
        """Fill gaps, scale with the training statistics, and add a channel marking missing glucose slots."""
        g0, h0 = np.atleast_2d(glucose), np.atleast_2d(heart_rate)
        g = np.stack([fill_gaps(row, self.glucose_mean) for row in g0])
        h = np.stack([fill_gaps(row, self.hr_mean) for row in h0])
        seq = np.stack([(g - self.glucose_mean) / self.glucose_std, (h - self.hr_mean) / self.hr_std,
                        np.isnan(g0).astype(float)], axis=1)  # fmt: skip
        p = np.atleast_2d(profile).astype(float)
        p = np.where(np.isnan(p), np.array(self.profile_mean), p)  # unknown -> training average
        p = (p - np.array(self.profile_mean)) / np.array(self.profile_std)
        return ModelInputs(sequence=seq.astype(np.float32), profile=p.astype(np.float32))

    def save(self, path: Path) -> None:
        """Write to JSON (shipped with the model)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Normaliser:
        """Read and re-check."""
        return cls.model_validate_json(path.read_text())
