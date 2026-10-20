"""End-to-end inference: load input -> deterministic checks -> model -> thresholded decision -> RiskCard."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger
from pydantic import ValidationError

from glimpse.config import GlimpseConfig
from glimpse.data.preprocess import CGMWindow, cgm_window
from glimpse.pipeline.checks import run_checks
from glimpse.schemas import AbstainReason, PreMealInput, RiskCard


class RiskModel(Protocol):
    """Interface every spike-risk model implements (placeholder now, trained CNN later)."""

    version: str

    def predict_proba(self, window: CGMWindow, inp: PreMealInput) -> float:
        """Return the probability of a large glucose spike."""
        ...


class PlaceholderTrendModel:
    """Stand-in until the trained sensor CNN exists: logistic function of the last-60-min glucose slope.

    It is NOT a trained model; it only lets the end-to-end interface, checks and abstention path run.
    """

    version = "placeholder-trend-0"

    def predict_proba(self, window: CGMWindow, inp: PreMealInput) -> float:
        """Map the recent glucose slope (mg/dL per min) to a probability."""
        last = window.values[-12:]
        ok = ~np.isnan(last)
        slope = float(np.polyfit(np.arange(12)[ok] * 5.0, last[ok], 1)[0]) if ok.sum() >= 3 else 0.0
        return 1.0 / (1.0 + math.exp(-(slope - 0.3) * 4.0))


def load_input(folder: str | Path) -> PreMealInput:
    """Read a sample folder: meal.json (meal_time, profile, optional photo), cgm.csv, optional heart_rate.csv."""
    folder = Path(folder)
    meta = json.loads((folder / "meal.json").read_text())
    with open(folder / "cgm.csv") as f:
        cgm = [{"timestamp": r["timestamp"], "glucose_mgdl": float(r["glucose_mgdl"])} for r in csv.DictReader(f)]
    hr = []
    if (folder / "heart_rate.csv").is_file():
        with open(folder / "heart_rate.csv") as f:
            hr = [{"timestamp": r["timestamp"], "bpm": float(r["bpm"])} for r in csv.DictReader(f)]
    photo = folder / meta["photo"] if meta.get("photo") else None
    return PreMealInput.model_validate(
        {"meal_time": meta["meal_time"], "photo_path": photo, "cgm": cgm, "heart_rate": hr, "profile": meta["profile"]}
    )


def _abstain(
    reasons: list[AbstainReason], cfg: GlimpseConfig, version: str, t0: float, p: float | None = None
) -> RiskCard:
    """Build a structured abstention."""
    return RiskCard(
        status="abstain",
        decision=None,
        spike_probability=p,
        reasons=reasons,
        message="Cannot judge this meal. General tips: eat vegetables and protein first, take a smaller carb "
        "portion, walk 10 minutes after eating. You can send this meal to your dietitian.",
        thresholds=(cfg.decision.low_risk_below, cfg.decision.caution_at_or_above),
        model_version=version,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


def predict(folder: str | Path, cfg: GlimpseConfig, model: RiskModel | None = None) -> RiskCard:
    """Run the full pipeline on one sample folder and always return a validated RiskCard (never raise on bad input)."""
    t0 = time.perf_counter()
    model = model or PlaceholderTrendModel()
    log = logger.bind(stage="predict", model=model.version, sample=str(folder))
    try:
        inp = load_input(folder)
    except (OSError, KeyError, ValueError, ValidationError) as e:
        log.warning("input rejected: {}", str(e).splitlines()[0])
        return _abstain([AbstainReason.INVALID_INPUT], cfg, model.version, t0)

    window = cgm_window(inp.cgm, inp.meal_time, cfg.pipeline.cgm_lookback_min)
    reasons = run_checks(inp, window, cfg.pipeline)
    if reasons:
        card = _abstain(reasons, cfg, model.version, t0)
        log.info("decision=abstain reasons={}", [r.value for r in reasons])
        return card

    p = model.predict_proba(window, inp)
    lo, hi = cfg.decision.low_risk_below, cfg.decision.caution_at_or_above
    if lo <= p < hi:
        log.info("decision=abstain (low confidence) p={:.3f} band=[{}, {})", p, lo, hi)
        return _abstain([AbstainReason.LOW_CONFIDENCE], cfg, model.version, t0, p)
    decision = "low_risk" if p < lo else "caution"
    card = RiskCard(
        status="answer",
        decision=decision,
        spike_probability=p,
        message="Low risk of a large glucose rise."
        if decision == "low_risk"
        else "Caution: a large glucose rise is likely.",
        thresholds=(lo, hi),
        model_version=model.version,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    log.info("decision={} p={:.3f} latency={:.1f} ms", decision, p, card.latency_ms)
    return card
