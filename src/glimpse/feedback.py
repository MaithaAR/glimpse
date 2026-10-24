"""Mocked feedback loop (intermediate stage): what happened after the meal, saved next to what the app said.

Two sources: the user ("ate it all / ate part of it / did not eat it", optionally the rise they saw), and the CGM
itself 2 hours later (`auto_rise`), which labels the meal with the same rule as the training data. Records are
validated (Pydantic) and appended as JSON lines to `prototype.feedback_file`; in the product they would queue for
review and, after checks, for retraining."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from glimpse.config import GlimpseConfig

Ate = Literal["ate it all", "ate part of it", "did not eat it"]


class FeedbackRecord(BaseModel):
    """One correction: the app's answer, what the user reports, and the label it implies."""

    model_config = ConfigDict(strict=True, extra="forbid")
    received_at: datetime
    source: Literal["user", "cgm_auto"]
    meal_time: str | None = None
    app_decision: Literal["low risk", "caution", "can't judge"]
    app_risk: float | None = Field(default=None, ge=0, le=1)
    ate: Ate | None = None
    observed_rise_mgdl: float | None = Field(default=None, ge=-300, le=400)
    feedback_label: Literal["spike", "no spike", "not eaten", "unknown"]


def label_from(ate: str | None, rise: float | None, threshold: float) -> str:
    """The label the feedback implies: not eaten wins; otherwise the rise decides; otherwise unknown."""
    if ate == "did not eat it":
        return "not eaten"
    if rise is None or math.isnan(rise):
        return "unknown"
    return "spike" if rise >= threshold else "no spike"


def auto_rise(times: list[datetime], values: list[float], meal_time: datetime, cfg: GlimpseConfig) -> float | None:
    """The CGM's own label 2 h later: peak within the horizon minus the mean of the 30 min before the meal
    (the training label's rule). None if either part has no readings."""
    lab = cfg.label
    before = [v for t, v in zip(times, values, strict=True)
              if meal_time - timedelta(minutes=lab.baseline_window_min) <= t <= meal_time]  # fmt: skip
    after = [v for t, v in zip(times, values, strict=True)
             if meal_time < t <= meal_time + timedelta(minutes=lab.horizon_min)]  # fmt: skip
    return max(after) - sum(before) / len(before) if before and after else None


def record_feedback(cfg: GlimpseConfig, answer: dict, ate: str | None, rise: float | None,
                    source: Literal["user", "cgm_auto"] = "user") -> FeedbackRecord:  # fmt: skip
    """Validate and append one feedback record; log it."""
    rise = None if rise is None or (isinstance(rise, float) and math.isnan(rise)) else float(rise)
    r = FeedbackRecord(received_at=datetime.now(), source=source, meal_time=answer.get("meal_time") or None,
                       app_decision=answer["decision"], app_risk=answer.get("risk"), ate=ate or None,
                       observed_rise_mgdl=rise,
                       feedback_label=label_from(ate, rise, cfg.label.spike_threshold_mgdl))  # fmt: skip
    path = Path(cfg.prototype.feedback_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(r.model_dump_json() + "\n")
    logger.bind(stage="feedback").info("feedback ({}): app said {}, label {}", source, r.app_decision,
                                       r.feedback_label)  # fmt: skip
    return r
