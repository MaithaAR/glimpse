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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import PROFILE
from glimpse.data.photos import PhotoError, eval_transform, open_photo
from glimpse.data.preprocess import passes_premeal_checks, resample_window
from glimpse.training.baselines import trend_features
from glimpse.training.final import FINAL, _photo_checkpoint, load_thresholds
from glimpse.training.photo import load_photo_bundle

AMENDED = Path("checkpoints/thresholds_amended.json")
OOD_REF = Path("checkpoints/photo_ood_reference.pt")
CARB_BINS = ["under 30 g", "30-59 g", "60-89 g", "90 g or more"]
PERSON_INPUTS = ["age", "bmi", "a1c_pct", "fasting_glucose_mgdl", "prior_spike_rate", "prior_rise"]


@dataclass
class MealInput:
    """Everything the app knows before the first bite."""

    meal_time: datetime
    glucose_times: list[datetime]
    glucose_mgdl: list[float]
    profile: dict[str, float]  # PERSON_INPUTS; prior_* may be NaN for a new user (filled from training people)
    photo: Path | None = None
    hr_times: list[datetime] = field(default_factory=list)
    hr_bpm: list[float] = field(default_factory=list)


@dataclass
class Answer:
    """What the user sees (decision + sentence + reason) and what was measured."""

    decision: str  # "low risk" | "caution" | "can't judge"
    sentence: str
    reason: str
    risk: float | None = None
    carbs_g: float | None = None
    carb_bin: str | None = None
    unfamiliar_photo_score: float | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


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
        torch.set_num_threads(min(cfg.training.num_threads, 4))
        self.cfg = cfg
        self.fusion = joblib.load(FINAL)["models"]["photo"]
        main = load_thresholds(AMENDED)["main"]
        self.lo, self.hi = main["low_risk_below"], main["caution_from"]
        self.model, _ = load_photo_bundle(cfg, _photo_checkpoint())
        self.model.eval()
        self.tf = eval_transform(cfg.photo.arch, cfg.photo.crop_pct)
        ref = torch.load(OOD_REF, weights_only=True)
        self.ref, self.ood_cut, self.k = ref["emb"].float(), float(ref["cut"]), int(ref["k"])

    @torch.no_grad()
    def embed_image(self, img) -> torch.Tensor:
        """PIL image -> the photo model's 768 features."""
        return self.model.embed(self.tf(img)[None])

    def unfamiliar_score(self, emb: torch.Tensor) -> float:
        """How far the photo is from the training photos (higher = more unusual)."""
        return float(knn_distance(self.ref, torch.nn.functional.normalize(emb, dim=1), self.k)[0])

    @torch.no_grad()
    def predict(self, m: MealInput) -> Answer:
        t0 = time.perf_counter()
        pc = self.cfg.pipeline

        def done(a: Answer) -> Answer:
            a.latency_ms = (time.perf_counter() - t0) * 1000
            return a

        g = resample_window(m.glucose_times, m.glucose_mgdl, m.meal_time, pc.cgm_lookback_min)
        h = resample_window(m.hr_times, m.hr_bpm, m.meal_time, pc.cgm_lookback_min)
        problem = window_problem(g, self.cfg)
        if problem:
            return done(Answer("can't judge", sentence("can't judge", None, problem), problem))
        if m.photo is None:
            r = "no meal photo was taken"
            return done(Answer("can't judge", sentence("can't judge", None, r), r))
        try:
            img = open_photo(m.photo, self.cfg.photo.min_side_px)
        except PhotoError as e:
            r = f"the photo cannot be used ({e})"
            return done(Answer("can't judge", sentence("can't judge", None, r), r))
        emb = self.embed_image(img)
        score = self.unfamiliar_score(emb)
        reg, logits = self.model.heads(emb)
        grams = float(torch.expm1(reg).clamp(min=0)[0])
        probs = torch.softmax(logits, 1)[0].numpy()
        base = {"carbs_g": grams, "carb_bin": CARB_BINS[int(probs.argmax())], "unfamiliar_photo_score": score}
        if score > self.ood_cut:
            r = "the photo looks unlike the meal photos the model learned from"
            return done(Answer("can't judge", sentence("can't judge", None, r), r, **base))
        ns = SimpleNamespace(glucose=g.values[None], heart_rate=h.values[None],
                             profile=profile_vector(m.profile, m.meal_time)[None])  # fmt: skip
        x = np.hstack([trend_features(ns, np.array([0])), [[np.log1p(grams), *probs[1:]]]])
        risk = float(self.fusion.predict_proba(x)[0, 1])
        dec = decision(risk, self.lo, self.hi)
        reason = {"low risk": f"risk {risk:.0%} is below the low-risk cut-off {self.lo:.0%}",
                  "caution": f"risk {risk:.0%} is at or above the caution cut-off {self.hi:.0%}",
                  "can't judge": f"the risk ({risk:.0%}) is borderline, between {self.lo:.0%} and "
                                 f"{self.hi:.0%}"}[dec]  # fmt: skip
        return done(Answer(dec, sentence(dec, risk, reason), reason, risk=risk, **base))
