"""The product's data contracts: what a meal submission must contain and what every answer carries.

Validated with strict Pydantic; JSON at the boundary. Report: the structured-output examples in §2."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CARB_BINS = ["under 30 g", "30-59 g", "60-89 g", "90 g or more"]
PERSON_INPUTS = ["age", "bmi", "a1c_pct", "fasting_glucose_mgdl", "prior_spike_rate", "prior_rise"]
Decision = Literal["low risk", "caution", "can't judge"]
PROFILE_RANGES = {"age": (18, 100), "bmi": (12, 70), "a1c_pct": (3, 15), "fasting_glucose_mgdl": (40, 400)}


class MealInput(BaseModel):
    """Everything the app knows before the first bite (validated on creation)."""

    model_config = ConfigDict(strict=True, extra="forbid")
    meal_time: datetime
    glucose_times: list[datetime]
    glucose_mgdl: list[float]
    profile: dict[str, float]  # PERSON_INPUTS; prior_* may be NaN for a new user (filled from training people)
    photo: Path | None = None
    hr_times: list[datetime] = Field(default_factory=list)
    hr_bpm: list[float] = Field(default_factory=list)

    @field_validator("profile")
    @classmethod
    def _profile_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        """Profile values must be physiologically plausible; a missing value (NaN) is filled later."""
        for key, (lo, hi) in PROFILE_RANGES.items():
            x = v.get(key)
            if x is not None and math.isfinite(x) and not lo <= x <= hi:
                raise ValueError(f"{key} = {x} is outside the plausible range {lo}-{hi}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> MealInput:
        """Times and values pair up; only the known profile inputs are accepted."""
        if len(self.glucose_times) != len(self.glucose_mgdl) or len(self.hr_times) != len(self.hr_bpm):
            raise ValueError("each reading needs a time and a value")
        unknown = set(self.profile) - set(PERSON_INPUTS)
        if unknown:
            raise ValueError(f"unknown profile inputs: {sorted(unknown)}")
        return self


class Answer(BaseModel):
    """What the user sees (decision + sentence + reason) and what was measured; JSON at the boundary."""

    model_config = ConfigDict(strict=True, extra="forbid")
    decision: Decision
    sentence: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    risk: float | None = Field(default=None, ge=0, le=1)
    # point estimate from the regression head; kept in results, not shown (the two heads were trained separately
    # and their agreement is not yet measured, so the user sees the bin head's range and confidence only)
    carbs_g: float | None = Field(default=None, ge=0)
    carb_bin: str | None = None
    carb_bin_prob: float | None = Field(default=None, ge=0, le=1)
    unfamiliar_photo_score: float | None = None
    latency_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _risk_when_answered(self) -> Answer:
        """An answer of 'low risk' or 'caution' must carry the risk it is based on."""
        if self.decision != "can't judge" and self.risk is None:
            raise ValueError("an answered meal needs a risk")
        if self.carb_bin is not None and self.carb_bin not in CARB_BINS:
            raise ValueError(f"unknown carb range {self.carb_bin}")
        return self
