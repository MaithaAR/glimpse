"""The GLIMPSE prototype (Step 6a): one meal in, one answer out.

Input: a before-meal photo, the CGM (and heart-rate) readings of the 2 hours before the meal, and a short profile
(age, BMI, HbA1c, fasting glucose, and the personal prior from the onboarding week).
Output: "low risk", "caution" or "can't judge", a plain-language sentence, the reason, the estimated carbs, and the
time it took. Rules run before the model, so a bad input gives "can't judge" with the reason, never a guess:
1. glucose data incomplete or out of the sensor's range (the same checks used to build the dataset);
2. no photo, or a photo that cannot be read or is too small;
3. a photo unlike the meal photos the model learned from (distance to the nearest training photos, measured in the
   photo model's own features; cut-off = 99th percentile of the training photos' own distances);
4. a risk between the two decision cut-offs (frozen before the test, see checkpoints/thresholds_amended.json).
It is decision support for a second look before eating, not a medical device."""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import joblib
import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import PROFILE
from glimpse.data.photos import PhotoError, eval_transform, open_photo
from glimpse.data.preprocess import passes_premeal_checks, resample_window
from glimpse.training.baselines import trend_features
from glimpse.training.final import FINAL, _photo_checkpoint, load_thresholds
from glimpse.training.photo import load_photo_bundle

CARB_BINS = ["under 30 g", "30-59 g", "60-89 g", "90 g or more"]
PERSON_INPUTS = ["age", "bmi", "a1c_pct", "fasting_glucose_mgdl", "prior_spike_rate", "prior_rise"]
Decision = Literal["low risk", "caution", "can't judge"]


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
    # and disagree on about half of the meals, so the user sees the bin head's range and confidence only)
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


def cant_judge(reason: str, **extra) -> Answer:
    """The structured 'can't judge' result (escalation: the user decides, or asks their care team)."""
    return Answer(decision="can't judge", sentence=sentence("can't judge", None, reason), reason=reason, **extra)


# ---------------------------------------------------------------- small pure rules (tested without any model)
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
        return (f"glucose data incomplete: {w.readings_last_30min} readings in the last 30 min (need "
                f"{pc.min_cgm_readings_last_30min}), {w.coverage:.0%} of the 2 h covered (need "
                f"{pc.min_cgm_coverage:.0%}), longest gap {w.max_gap_min:.0f} min "
                f"(max {pc.max_cgm_gap_min:.0f})")  # fmt: skip
    return None


def decision(risk: float, lo: float, hi: float) -> str:
    """Frozen cut-offs: low risk below lo, caution from hi, can't judge in between."""
    return "low risk" if risk < lo else "caution" if risk >= hi else "can't judge"


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


# ---------------------------------------------------------------- the app
class Glimpse:
    """Loads the frozen 28-person pipeline once; `predict` answers one meal."""

    def __init__(self, cfg: GlimpseConfig):
        """Load the fusion model, the frozen cut-offs (checked against the config), the photo model and the
        unfamiliar-photo reference, all from local files."""
        torch.set_num_threads(min(cfg.training.num_threads, 4))
        self.cfg, pc = cfg, cfg.prototype
        self.fusion = joblib.load(FINAL)["models"]["photo"]
        frozen = load_thresholds(Path(pc.thresholds_file))["main"]
        if max(abs(frozen["low_risk_below"] - pc.low_risk_below), abs(frozen["caution_from"] - pc.caution_from)) > 1e-9:
            raise ValueError("prototype cut-offs in the config differ from the frozen amendment file")
        self.lo, self.hi = pc.low_risk_below, pc.caution_from
        self.model, _ = load_photo_bundle(cfg, _photo_checkpoint())
        self.model.eval()
        self.tf = eval_transform(cfg.photo.arch, cfg.photo.crop_pct)
        ref = torch.load(pc.ood_reference_file, weights_only=True)
        self.ref, self.ood_cut, self.k = ref["emb"].float(), float(ref["cut"]), int(ref["k"])
        self.log = logger.bind(stage="predict")

    @torch.no_grad()
    def embed_image(self, img) -> torch.Tensor:
        """PIL image -> the photo model's 768 features."""
        return self.model.embed(self.tf(img)[None])

    def unfamiliar_score(self, emb: torch.Tensor) -> float:
        """How far the photo is from the training photos (higher = more unusual)."""
        return float(knn_distance(self.ref, torch.nn.functional.normalize(emb, dim=1), self.k)[0])

    @torch.no_grad()
    def predict(self, m: MealInput) -> Answer:
        """One meal -> one validated Answer. Rules first; the models only see inputs that passed them."""
        t0 = time.perf_counter()
        a = self._answer(m)
        a.latency_ms = (time.perf_counter() - t0) * 1000
        msg = "{} ({:.0f} ms): {}"
        (self.log.info if a.decision == "can't judge" else self.log.debug)(msg, a.decision, a.latency_ms, a.reason)
        return a

    def _answer(self, m: MealInput) -> Answer:
        """The decision itself (timing and logging are added by `predict`)."""
        pc = self.cfg.pipeline
        g = resample_window(m.glucose_times, m.glucose_mgdl, m.meal_time, pc.cgm_lookback_min)
        h = resample_window(m.hr_times, m.hr_bpm, m.meal_time, pc.cgm_lookback_min)
        problem = window_problem(g, self.cfg)
        if problem:
            return cant_judge(problem)
        if m.photo is None:
            return cant_judge("no meal photo was taken")
        try:
            img = open_photo(m.photo, self.cfg.photo.min_side_px)
        except PhotoError as e:
            return cant_judge(f"the photo cannot be used ({e})")
        emb = self.embed_image(img)
        score = self.unfamiliar_score(emb)
        if score > self.ood_cut:  # no carb guess for a photo that does not look like a meal
            r = "the photo looks unlike the meal photos the model learned from"
            return cant_judge(r, unfamiliar_photo_score=score)
        reg, logits = self.model.heads(emb)
        grams = float(torch.expm1(reg).clamp(min=0)[0])
        probs = torch.softmax(logits, 1)[0].numpy()
        base = {"carbs_g": grams, "carb_bin": CARB_BINS[int(probs.argmax())], "carb_bin_prob": float(probs.max()),
                "unfamiliar_photo_score": score}  # fmt: skip
        ns = SimpleNamespace(glucose=g.values[None], heart_rate=h.values[None],
                             profile=profile_vector(m.profile, m.meal_time)[None])  # fmt: skip
        x = np.hstack([trend_features(ns, np.array([0])), [[np.log1p(grams), *probs[1:]]]])
        risk = float(self.fusion.predict_proba(x)[0, 1])
        dec = decision(risk, self.lo, self.hi)
        if dec == "can't judge":
            return cant_judge(f"the risk ({risk:.0%}) is borderline, between {self.lo:.0%} and {self.hi:.0%}",
                              risk=risk, **base)  # fmt: skip
        reason = (f"risk {risk:.0%} is below the low-risk cut-off {self.lo:.0%}" if dec == "low risk"
                  else f"risk {risk:.0%} is at or above the caution cut-off {self.hi:.0%}")  # fmt: skip
        return Answer(decision=dec, sentence=sentence(dec, risk, reason), reason=reason, risk=risk, **base)


def read_cgm_csv(path: str | Path) -> tuple[list[datetime], list[float], list[datetime], list[float]]:
    """A sensor export as the app accepts it: columns `timestamp`, `glucose_mgdl` and optionally `hr_bpm`."""
    import pandas as pd

    d = pd.read_csv(path)
    missing = {"timestamp", "glucose_mgdl"} - set(d.columns)
    if missing:
        raise ValueError(f"the CSV needs the columns timestamp and glucose_mgdl (missing: {sorted(missing)})")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    g = d.dropna(subset=["glucose_mgdl"])
    h = d.dropna(subset=["hr_bpm"]) if "hr_bpm" in d else d.iloc[0:0]
    return (list(g.timestamp.dt.to_pydatetime()), g.glucose_mgdl.astype(float).tolist(),
            list(h.timestamp.dt.to_pydatetime()), h.hr_bpm.astype(float).tolist() if len(h) else [])  # fmt: skip


def answer_from_files(app: Glimpse, cgm: str | Path | None, photo: str | Path | None, meal_time: str | None,
                      profile: dict[str, float | None]) -> Answer:  # fmt: skip
    """The app's entry point for user files: anything malformed, empty or unsupported becomes a structured
    'can't judge' with the reason, never a crash."""
    try:
        if cgm is None:
            raise ValueError("no CGM export was given")
        gt, gv, ht, hv = read_cgm_csv(cgm)
        if not gt:
            raise ValueError("the CGM export has no glucose readings")
        when = datetime.fromisoformat(meal_time) if meal_time else max(gt)
        prof = {k: float("nan") if v is None else float(v) for k, v in profile.items()}
        m = MealInput(meal_time=when, glucose_times=gt, glucose_mgdl=gv, hr_times=ht, hr_bpm=hv, profile=prof,
                      photo=Path(photo) if photo else None)  # fmt: skip
    except (ValueError, OSError, TypeError) as e:
        a = cant_judge(f"unsupported input ({e})")
        logger.bind(stage="predict").warning("can't judge: {}", a.reason)
        return a
    return app.predict(m)
