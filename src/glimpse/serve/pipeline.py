"""The one scoring path (design decision D2): rules -> photo estimate -> fusion -> cut-offs -> Answer.

`just demo`, `just predict`, the app and the folder command all answer through Glimpse here; evaluation joins in
the splits-v2 step via the precomputed-table seam. Report: system diagram in §2."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from loguru import logger

from glimpse.config import GlimpseConfig
from glimpse.data.photos import PhotoError, eval_transform, open_photo
from glimpse.data.preprocess import resample_window
from glimpse.serve.rules import (
    cant_judge,
    decision,
    knn_distance,
    photo_problem,
    profile_vector,
    sentence,
    window_problem,
)
from glimpse.serve.schemas import CARB_BINS, Answer, MealInput
from glimpse.training.baselines import trend_features
from glimpse.training.final import FINAL, _photo_checkpoint, load_thresholds
from glimpse.training.photo import load_photo_bundle


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
        blank = photo_problem(img, self.cfg.prototype.min_photo_contrast)
        if blank:
            return cant_judge(blank)
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
