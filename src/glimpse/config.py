"""Typed, validated settings. configs/default.yaml is read once at startup and checked against these models:
unknown keys, missing keys, wrong types and impossible values all stop the program with a clear message."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Prob = Annotated[float, Field(ge=0, le=1)]
CostPair = Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=2, max_length=2)]  # (if spike, if no spike)
Group = Literal["healthy", "prediabetes", "T2D"]


class _Strict(BaseModel):
    """Base for every settings block: no type guessing, no unknown keys, read-only after loading."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class PathsCfg(_Strict):
    """Where generated files go."""

    outputs: str
    logs: str


class DataCfg(_Strict):
    """Where the downloaded datasets live."""

    cgmacros_root: str
    nutrition5k_root: str
    reference_sensor: Literal["dexcom", "libre"]


class LabelCfg(_Strict):
    """Definition of a spike: peak rise above the pre-meal baseline within the horizon."""

    spike_threshold_mgdl: float = Field(gt=0)
    sensitivity_thresholds: list[float] = Field(min_length=1)
    horizon_min: int = Field(gt=0)
    baseline_window_min: int = Field(gt=0)


class StrictWindowCfg(_Strict):
    """Tighter window rules used in a sensitivity check."""

    min_post_coverage: Prob
    max_gap_min: int = Field(gt=0)


class WindowRulesCfg(_Strict):
    """When a meal's CGM window is good enough to label."""

    min_baseline_readings: int = Field(ge=1)
    min_post_coverage: Prob
    max_gap_min: int = Field(gt=0)
    strict: StrictWindowCfg


class MealsCfg(_Strict):
    """Meal-level exclusions."""

    min_share_eaten_pct: float = Field(ge=0, le=100)
    exclude_next_meal_within_min: int = Field(ge=0)


class OnboardingCfg(_Strict):
    """The first hours of each person build the personal prior."""

    hours: int = Field(gt=0)
    min_meals_for_personal_prior: int = Field(ge=1)


class CohortCfg(_Strict):
    """Which groups the product is for (results are reported on these)."""

    target_groups: list[Group] = Field(min_length=1)


class CvCfg(_Strict):
    """Person-grouped cross-validation settings."""

    outer_folds: int = Field(ge=2)
    inner_folds: int = Field(ge=2)
    seeds: list[int] = Field(min_length=1)


class CostsCfg(_Strict):
    """Cost of each answer when the meal does / does not spike (assumptions, varied in a sensitivity check)."""

    low_risk: CostPair
    caution: CostPair
    abstain: CostPair
    sweep_missed_spike: list[float] = Field(min_length=1)
    sweep_abstain_multiplier: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _missed_spike_is_worst(self) -> CostsCfg:
        """Saying 'low risk' before a spike must cost more than a caution would; a needless caution costs more."""
        if self.low_risk[0] <= self.caution[0] or self.caution[1] <= self.low_risk[1]:
            raise ValueError("costs: a missed spike must cost more than a caution, and a needless caution more than 0")
        return self


class UsefulnessCfg(_Strict):
    """Day-to-day usefulness targets."""

    min_cost_reduction: Prob
    min_cleared: Prob
    max_missed_among_cleared: Prob
    max_abstain: Prob


class TargetsCfg(_Strict):
    """Success targets fixed before seeing results."""

    rq1_noninferiority_margin: float = Field(gt=0, lt=1)
    ece_max: Prob
    ece_bins: int = Field(ge=2, le=50)
    conformal_nominal: Prob
    usefulness: UsefulnessCfg
    latency_p95_ms: float = Field(gt=0)


class GlimpseConfig(_Strict):
    """All settings of the project, mirroring configs/default.yaml."""

    seed: int = Field(ge=0)
    paths: PathsCfg
    data: DataCfg
    label: LabelCfg
    window_rules: WindowRulesCfg
    meals: MealsCfg
    onboarding: OnboardingCfg
    cohort: CohortCfg
    cv: CvCfg
    costs: CostsCfg
    targets: TargetsCfg


def load_config(path: str | Path = "configs/default.yaml") -> GlimpseConfig:
    """Read the YAML file and validate it; raises pydantic.ValidationError listing every problem found."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return GlimpseConfig.model_validate(raw)
