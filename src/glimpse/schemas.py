"""Pydantic data contracts shared by every pipeline stage (inputs, samples, outputs, evaluation results)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Contract(BaseModel):
    """Strict base: no type coercion, no unknown fields."""

    model_config = ConfigDict(strict=True, extra="forbid")


# ---------------------------------------------------------------- inputs (what the user's phone would send)


class GlucoseReading(_Contract):
    """One CGM reading."""

    model_config = ConfigDict(strict=False, extra="forbid")  # timestamps arrive as ISO strings from CSV/JSON
    timestamp: datetime
    glucose_mgdl: float = Field(gt=0, lt=1000)


class HeartRateReading(_Contract):
    """One heart-rate reading from the wearable."""

    model_config = ConfigDict(strict=False, extra="forbid")
    timestamp: datetime
    bpm: float = Field(gt=20, lt=250)


class Profile(_Contract):
    """User profile entered at onboarding; labs are optional."""

    age: int = Field(ge=18, le=100)
    bmi: float = Field(ge=12, le=70)
    a1c_pct: float | None = Field(default=None, ge=3.5, le=15)
    fasting_glucose_mgdl: float | None = Field(default=None, ge=40, le=400)


class PreMealInput(_Contract):
    """Everything the product receives at the decision moment, before the first bite."""

    model_config = ConfigDict(strict=False, extra="forbid")
    meal_time: datetime
    photo_path: Path | None = None
    cgm: list[GlucoseReading]
    heart_rate: list[HeartRateReading] = Field(default_factory=list)
    profile: Profile

    @field_validator("cgm")
    @classmethod
    def _cgm_sorted_unique(cls, v: list[GlucoseReading]) -> list[GlucoseReading]:
        """CGM readings must be in time order without duplicate timestamps."""
        ts = [r.timestamp for r in v]
        if ts != sorted(ts) or len(set(ts)) != len(ts):
            raise ValueError("CGM readings must be sorted by time with unique timestamps")
        return v


# ---------------------------------------------------------------- training sample (one labelled meal)


class MealSample(_Contract):
    """One labelled meal used for training/evaluation (pre-meal features only as inputs)."""

    model_config = ConfigDict(strict=False, extra="forbid")
    person_id: int = Field(ge=1)
    meal_time: datetime
    group: Literal["healthy", "prediabetes", "T2D"]
    photo_relpath: str | None
    baseline_mgdl: float = Field(gt=0, lt=1000)
    rise_mgdl: float
    spike: Literal[0, 1]
    carbs_served_g: float | None = Field(default=None, ge=0)
    onboarding: bool  # outcome window ends inside the person's first 72 h (feeds the personal prior)
    isolated: bool  # post-onboarding and no further meal within 2 h (primary evaluation population)
    premeal_ok: bool  # passes the same pre-meal sensor checks the product applies at inference

    @field_validator("rise_mgdl")
    @classmethod
    def _plausible_rise(cls, v: float) -> float:
        """A 2-hour rise outside +/-400 mg/dL is a sensor artefact, not physiology."""
        if not -400 < v < 400:
            raise ValueError(f"implausible glucose rise {v}")
        return v


# ---------------------------------------------------------------- outputs


class AbstainReason(StrEnum):
    """Why the product declined to give a risk."""

    INVALID_INPUT = "invalid_input"
    MISSING_PHOTO = "missing_photo"
    UNREADABLE_PHOTO = "unreadable_photo"
    INSUFFICIENT_CGM = "insufficient_cgm"
    CGM_GAP = "cgm_gap"
    GLUCOSE_OUT_OF_RANGE = "glucose_out_of_range"
    LOW_CONFIDENCE = "low_confidence"


class RiskCard(_Contract):
    """The structured answer returned to the user (or the abstention, with explicit reasons)."""

    status: Literal["answer", "abstain"]
    decision: Literal["low_risk", "caution"] | None
    spike_probability: float | None = Field(default=None, ge=0, le=1)
    carb_estimate_g: float | None = Field(default=None, ge=0)
    reasons: list[AbstainReason] = Field(default_factory=list)
    message: str
    thresholds: tuple[float, float]
    model_version: str
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> RiskCard:
        """An answer carries a decision and no reasons; an abstention carries reasons and no decision."""
        if self.status == "answer" and (self.decision is None or self.reasons):
            raise ValueError("an answer needs a decision and no abstain reasons")
        if self.status == "abstain" and (self.decision is not None or not self.reasons):
            raise ValueError("an abstention needs at least one reason and no decision")
        return self


# ---------------------------------------------------------------- evaluation


class EvalResult(_Contract):
    """One model's metrics on one split."""

    model_name: str
    split: Literal["validation", "test", "cv"]
    n_meals: int = Field(ge=0)
    n_people: int = Field(ge=0)
    auroc: float = Field(ge=0, le=1)
    auroc_ci: tuple[float, float]
    brier: float = Field(ge=0, le=1)
    ece: float = Field(ge=0, le=1)
    abstention_rate: float = Field(ge=0, le=1)
    latency_ms_p95: float | None = Field(default=None, ge=0)
