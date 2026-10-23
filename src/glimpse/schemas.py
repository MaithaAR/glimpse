"""Data contracts: every record passed between pipeline stages is a validated Pydantic model, not a loose dict."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MealRecord(BaseModel):
    """One labelled meal from the meal table. Rows that break these rules are rejected (and logged), not kept."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)  # extra columns (features) are allowed

    sid: int = Field(ge=1, description="participant id")
    t: datetime = Field(description="meal time")
    group: Literal["healthy", "prediabetes", "T2D"]
    base: float = Field(gt=20, lt=600, description="mean glucose in the 30 min before the meal, mg/dL")
    rise: float = Field(gt=-300, lt=600, description="peak rise in the 2 h after the meal, mg/dL")
    coverage: float = Field(ge=0, le=1, description="share of post-meal 5-min slots with a reading")
    share: float = Field(gt=0, le=100, description="percentage of the plate eaten")
    spike50: Literal[0, 1] = Field(description="label: rise >= 50 mg/dL")
    carbs_served: float | None = Field(default=None, ge=0, description="label for the photo model, grams")
    photo: str | None = None
    usable: bool
    onboarding: bool
    post_onboarding: bool
    next_meal_within: bool

    @model_validator(mode="after")
    def _label_matches_rise(self) -> MealRecord:
        """The label must agree with the measured rise (catches a mixed-up column)."""
        if self.spike50 != int(self.rise >= 50):
            raise ValueError("spike50 does not match rise")
        return self


class MealSample(BaseModel):
    """One model-ready meal: identity, label, carb label and quality flags. Inputs live in aligned arrays."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    person_id: int = Field(ge=1)
    meal_time: datetime
    group: Literal["healthy", "prediabetes", "T2D"]
    photo_relpath: str | None
    baseline_mgdl: float = Field(gt=20, lt=600)
    rise_mgdl: float = Field(gt=-300, lt=600)
    spike: Literal[0, 1]
    carbs_served_g: float | None = Field(default=None, ge=0)
    onboarding: bool  # outcome window ends in the person's first 72 h (used for the personal prior, not scored)
    isolated: bool  # after onboarding and no other meal within 2 h: the meals that are scored
    premeal_ok: bool  # passes the product's pre-meal sensor checks
