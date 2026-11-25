"""Deterministic rules, in plain functions tested without any model: the pre-model input checks, the
cut-off decision, the user-facing wording, and the unfamiliar-photo distance. Report: §4 and docs/rule_ledger.md."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import torch

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import PROFILE
from glimpse.data.preprocess import passes_premeal_checks
from glimpse.serve.schemas import Answer


def cant_judge(reason: str, **extra) -> Answer:
    """The structured 'can't judge' result (escalation: the user decides, or asks their care team)."""
    return Answer(decision="can't judge", sentence=sentence("can't judge", None, reason), reason=reason, **extra)


def profile_vector(profile: dict[str, float], meal_time: datetime) -> np.ndarray:
    """The 8 profile inputs in the model's order; the meal hour becomes a point on a 24-hour circle."""
    hour = meal_time.hour + meal_time.minute / 60
    vals = {**profile, "hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24)}
    return np.array([float(vals.get(k, float("nan"))) for k in PROFILE])


def window_problem(w, cfg: GlimpseConfig) -> str | None:
    """Why the glucose window cannot be trusted (None if it is fine): the product's rules before any model."""
    pc = cfg.pipeline
    lo, hi = pc.glucose_range_mgdl
    seen = w.values[~np.isnan(w.values)]
    if seen.size and (seen.min() < lo or seen.max() > hi):
        return f"a glucose reading is outside the sensor's range ({lo}-{hi} mg/dL)"
    if not passes_premeal_checks(w, pc.min_cgm_readings_last_30min, pc.min_cgm_coverage, pc.max_cgm_gap_min):
        failed = []  # name only the criteria that failed (26 Nov: passing ones read as failures)
        if w.readings_last_30min < pc.min_cgm_readings_last_30min:
            failed.append(
                f"{w.readings_last_30min} readings in the last 30 min (need {pc.min_cgm_readings_last_30min})"
            )
        if w.coverage < pc.min_cgm_coverage:
            failed.append(f"{w.coverage:.0%} of the 2 h covered (need {pc.min_cgm_coverage:.0%})")
        if w.max_gap_min > pc.max_cgm_gap_min:
            failed.append(f"a gap of {w.max_gap_min:.0f} min (max {pc.max_cgm_gap_min:.0f})")
        return "glucose data incomplete: " + ", ".join(failed or ["the pre-meal checks failed"])
    return None


def photo_problem(img, min_contrast: float) -> str | None:
    """A deterministic check before the photo model: a near-uniform picture (blank, covered lens, plain wall) has
    almost no grey-level spread, so there is no meal to read. Returns the reason, or None if the photo is usable."""
    grey = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
    if float(grey.std()) < min_contrast:
        spread = float(grey.std())
        return f"the photo is almost blank (grey-level spread {spread:.1f}, need {min_contrast:.0f}); no meal visible"
    return None


def decision(risk: float, lo: float, hi: float) -> str:
    """Frozen cut-offs: low risk below lo, caution from hi, can't judge in between."""
    return "low risk" if risk < lo else "caution" if risk >= hi else "can't judge"


def decision_reason(dec: str, risk: float, lo: float, hi: float) -> str:
    """Why an answered meal got its decision. The risk prints at one decimal so it never reads as equal to
    the cut-off it was compared with ("risk 15% is below the low-risk cut-off 15%"; R11 addition, 26 Nov)."""
    if dec == "low risk":
        digits = next(d for d in (1, 2, 3, 4) if round(risk * 100, d) < lo * 100 or d == 4)
        return f"risk {risk:.{digits}%} is below the low-risk cut-off {lo:.0%}"
    return f"risk {risk:.1%} is at or above the caution cut-off {hi:.0%}"


PRIORS = ("prior_spike_rate", "prior_rise")


def no_basis_to_clear(dec: str, history, profile: dict[str, float]) -> str | None:
    """Rule R13 (26 Nov): a low-risk answer needs a personal basis, either earlier labelled meals (history)
    or onboarding priors. Without both the risk is the population calibration alone, which on development
    cleared 11% of meals with 16% of them spiking; such a meal is not cleared but referred, with this reason.
    Returns the reason, or None when the rule does not apply. Changes no risk, weight or cut-off."""
    if dec != "low risk" or history:
        return None
    if all(math.isfinite(float(profile.get(k, float("nan")))) for k in PRIORS):
        return None
    return (
        "not cleared: no earlier labelled meals and no onboarding priors, so the risk rests on the "
        "population alone (rule R13); add a history.csv or the onboarding priors, or ask a dietitian"
    )


def sentence(dec: str, risk: float | None, reason: str) -> str:
    """Plain words for the user."""
    if dec == "low risk":
        return f"Unlikely to spike: about {risk:.0%} chance this meal raises your glucose by 50+ mg/dL within 2 h."
    if dec == "caution":
        return (f"Caution: about {risk:.0%} chance this meal raises your glucose by 50+ mg/dL within 2 h. "
                "A smaller carb portion or a short walk afterwards may help.")  # fmt: skip
    return f"Can't judge this meal: {reason}."


def knn_distance(ref: torch.Tensor, q: torch.Tensor, k: int) -> torch.Tensor:
    """Mean cosine distance from each query to its k nearest reference photos (inputs L2-normalised)."""
    d = 1 - q @ ref.T
    return d.topk(k, dim=1, largest=False).values.mean(1)


def leave_one_out_distances(ref: torch.Tensor, k: int) -> torch.Tensor:
    """Each training photo's distance to its k nearest OTHER training photos (the 'normal' range)."""
    d = 1 - ref @ ref.T
    d.fill_diagonal_(float("inf"))
    return d.topk(k, dim=1, largest=False).values.mean(1)
