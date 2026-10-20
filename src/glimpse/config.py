"""Typed, validated project configuration loaded from YAML (fails fast on unknown keys or invalid values)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    """Base model: unknown keys are errors and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    """Where generated artefacts and caches live (relative to the repo root)."""

    outputs: Path
    logs: Path
    checkpoints: Path
    weights_cache: Path
    meals_table: Path
    dataset: Path
    splits: Path
    normaliser: Path


class DataCfg(_Strict):
    """Raw data locations."""

    cgmacros_root: Path
    nutrition5k_root: Path
    reference_sensor: Literal["dexcom", "libre"]


class LabelCfg(_Strict):
    """Spike label definition."""

    spike_threshold_mgdl: float = Field(gt=0, le=200)
    sensitivity_thresholds: list[float]
    horizon_min: int = Field(gt=0, le=360)
    baseline_window_min: int = Field(gt=0, le=120)


class StrictWindow(_Strict):
    """Stricter window rule used in sensitivity analyses."""

    min_post_coverage: float = Field(ge=0, le=1)
    max_gap_min: int = Field(gt=0)


class WindowRules(_Strict):
    """When a post-meal glucose window is usable for labelling."""

    min_baseline_readings: int = Field(ge=1)
    min_post_coverage: float = Field(ge=0, le=1)
    max_gap_min: int = Field(gt=0)
    strict: StrictWindow


class MealsCfg(_Strict):
    """Meal-level inclusion rules."""

    min_share_eaten_pct: float = Field(ge=0, le=100)
    exclude_next_meal_within_min: int = Field(ge=0)


class OnboardingCfg(_Strict):
    """Personal-prior onboarding period."""

    hours: int = Field(gt=0)
    min_meals_for_personal_prior: int = Field(ge=1)


class CohortCfg(_Strict):
    """Which A1c groups form the target cohort."""

    target_groups: list[Literal["healthy", "prediabetes", "T2D"]]


class CVCfg(_Strict):
    """Person-grouped cross-validation settings."""

    outer_folds: int = Field(ge=2)
    inner_folds: int = Field(ge=2)
    seeds: list[int]


class CostsCfg(_Strict):
    """Assumed decision costs as (spike, no_spike) pairs."""

    low_risk: tuple[float, float]
    caution: tuple[float, float]
    abstain: tuple[float, float]
    sweep_missed_spike: list[float]
    sweep_abstain_multiplier: list[float]


class UsefulnessTargets(_Strict):
    """Operational usefulness targets from the proposal."""

    min_cost_reduction: float
    min_cleared: float
    max_missed_among_cleared: float
    max_abstain: float


class TargetsCfg(_Strict):
    """Proposal success targets."""

    rq1_noninferiority_margin: float
    ece_max: float = Field(gt=0, lt=1)
    ece_bins: int = Field(ge=2, le=50)
    conformal_nominal: float = Field(gt=0, lt=1)
    usefulness: UsefulnessTargets
    latency_p95_ms: float = Field(gt=0)


class PipelineCfg(_Strict):
    """Deterministic input checks applied before any model runs."""

    cgm_lookback_min: int = Field(gt=0, le=360)
    min_cgm_readings_last_30min: int = Field(ge=1)
    min_cgm_coverage: float = Field(ge=0, le=1)
    max_cgm_gap_min: int = Field(gt=0)
    glucose_range_mgdl: tuple[float, float]
    min_photo_side_px: int = Field(ge=8)


class DecisionCfg(_Strict):
    """Probability thresholds that map spike risk to low risk / abstain / caution."""

    low_risk_below: float = Field(ge=0, le=1)
    caution_at_or_above: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> DecisionCfg:
        """The abstain band must not be inverted."""
        if self.low_risk_below > self.caution_at_or_above:
            raise ValueError("decision.low_risk_below must be <= decision.caution_at_or_above")
        return self


class ScratchTrainCfg(_Strict):
    """Hyperparameters of the from-scratch sensor CNN."""

    channels: int = Field(ge=1, le=256)
    hidden: int = Field(ge=1, le=1024)
    dropout: float = Field(ge=0, lt=1)
    lr: float = Field(gt=0, lt=1)
    weight_decay: float = Field(ge=0, lt=1)
    batch_size: int = Field(ge=1, le=4096)
    max_epochs: int = Field(ge=1, le=1000)
    warmup_epochs: int = Field(ge=0)
    patience: int = Field(ge=1)
    cv_folds: int = Field(ge=2, le=10)
    augment: bool


class TrainingCfg(_Strict):
    """Training settings shared by both models."""

    num_threads: int = Field(ge=1, le=64)
    tensorboard_dir: Path
    scratch: ScratchTrainCfg


class GlimpseConfig(_Strict):
    """Complete project configuration."""

    seed: int
    device: Literal["cpu", "mps", "cuda"]
    paths: Paths
    data: DataCfg
    label: LabelCfg
    window_rules: WindowRules
    meals: MealsCfg
    onboarding: OnboardingCfg
    cohort: CohortCfg
    cv: CVCfg
    costs: CostsCfg
    targets: TargetsCfg
    pipeline: PipelineCfg
    decision: DecisionCfg
    training: TrainingCfg


def load_config(path: str | Path = "configs/default.yaml") -> GlimpseConfig:
    """Load and validate the YAML configuration; raises pydantic.ValidationError with a clear message if invalid."""
    with open(path) as f:
        return GlimpseConfig.model_validate(yaml.safe_load(f))
