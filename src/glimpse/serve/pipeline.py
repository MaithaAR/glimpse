"""The one scoring path (design decision D2): rules -> carb estimate -> fusion -> cut-offs -> Answer.

`just demo`, `just predict`, the app and the folder command answer through `Pipeline` with the live photo model;
evaluation joins via a precomputed per-fold estimate table through the same decision code. Tier 2 (confirm-carbs,
D5): when the photo cannot be used but the user typed a carb estimate, the risk is still produced from it.
Report: system diagram in §2; the parity test in tests/test_serve.py is the evaluation-equals-deployment proof."""

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
from glimpse.serve.schemas import CARB_BINS, Answer, MealInput, PhotoEstimate
from glimpse.training.baselines import trend_features
from glimpse.training.final import FINAL, _photo_checkpoint, load_thresholds
from glimpse.training.photo import load_photo_bundle


def _bin_onehot(grams: float) -> list[float]:
    """The carb range a gram amount falls in, as one-hot probabilities over the four ranges."""
    idx = 0 if grams < 30 else 1 if grams < 60 else 2 if grams < 90 else 3
    return [1.0 if i == idx else 0.0 for i in range(4)]


class Pipeline:
    """Loads the frozen 28-person pipeline once; `answer` scores one meal.

    `photo_table` switches the carb-estimate source from the live photo model to precomputed per-fold estimates
    (evaluation); everything from the rules to the Answer is the same code either way."""

    def __init__(self, cfg: GlimpseConfig, photo_table: dict[int, PhotoEstimate] | None = None):
        """Load the fusion model, the frozen cut-offs (checked against the config) and, in live mode, the photo
        model and the unfamiliar-photo reference, all from local files."""
        torch.set_num_threads(min(cfg.training.num_threads, 4))
        self.cfg, pc = cfg, cfg.prototype
        self.fusion = joblib.load(FINAL)["models"]["photo"]
        frozen_all = load_thresholds(Path(pc.thresholds_file))
        frozen = frozen_all["main"]
        if max(abs(frozen["low_risk_below"] - pc.low_risk_below), abs(frozen["caution_from"] - pc.caution_from)) > 1e-9:
            raise ValueError("prototype cut-offs in the config differ from the frozen amendment file")
        self.lo, self.hi = pc.low_risk_below, pc.caution_from
        self.table = photo_table
        self.model_versions = {"fusion": FINAL.name, "thresholds": str(frozen_all.get("sha256", ""))[:12]}
        if photo_table is None:
            self.model, _ = load_photo_bundle(cfg, _photo_checkpoint())
            self.model.eval()
            self.tf = eval_transform(cfg.photo.arch, cfg.photo.crop_pct)
            ref = torch.load(pc.ood_reference_file, weights_only=True)
            self.ref, self.ood_cut, self.k = ref["emb"].float(), float(ref["cut"]), int(ref["k"])
            self.model_versions["photo"] = _photo_checkpoint().name
        else:
            self.model_versions["photo"] = "precomputed-table"
        self.log = logger.bind(stage="predict")

    @torch.no_grad()
    def embed_image(self, img) -> torch.Tensor:
        """PIL image -> the photo model's 768 features."""
        return self.model.embed(self.tf(img)[None])

    def unfamiliar_score(self, emb: torch.Tensor) -> float:
        """How far the photo is from the training photos (higher = more unusual)."""
        return float(knn_distance(self.ref, torch.nn.functional.normalize(emb, dim=1), self.k)[0])

    @torch.no_grad()
    def photo_estimate(self, m: MealInput, key: int | None = None) -> PhotoEstimate | tuple[str, str, dict]:
        """The carb estimate for a meal, or (failed_check, reason, extra Answer fields) when the photo is unusable."""
        if self.table is not None:
            if key in self.table:
                return self.table[key]
            return ("photo_present", "no precomputed photo estimate for this meal", {})
        if m.photo is None:
            return ("photo_present", "no meal photo was taken", {})
        try:
            img = open_photo(m.photo, self.cfg.photo.min_side_px)
        except PhotoError as e:
            return ("photo_readable", f"the photo cannot be used ({e})", {})
        blank = photo_problem(img, self.cfg.prototype.min_photo_contrast)
        if blank:
            return ("photo_not_blank", blank, {})
        emb = self.embed_image(img)
        score = self.unfamiliar_score(emb)
        if score > self.ood_cut:  # no model carb guess for a photo that does not look like a meal
            r = "the photo looks unlike the meal photos the model learned from"
            return ("photo_familiar", r, {"unfamiliar_photo_score": score})
        reg, logits = self.model.heads(emb)
        grams = float(torch.expm1(reg).clamp(min=0)[0])
        probs = torch.softmax(logits, 1)[0].tolist()
        return PhotoEstimate(grams=grams, bin_probs=probs, unfamiliar_score=score)

    def answer(self, m: MealInput, key: int | None = None) -> Answer:
        """One meal -> one validated Answer. Rules first; the models only see inputs that passed them."""
        t0 = time.perf_counter()
        a = self._answer(m, key)
        a.latency_ms = (time.perf_counter() - t0) * 1000
        msg = "{} ({:.0f} ms): {}"
        (self.log.info if a.decision == "can't judge" else self.log.debug)(msg, a.decision, a.latency_ms, a.reason)
        return a

    def _answer(self, m: MealInput, key: int | None = None) -> Answer:
        """The decision itself (timing and logging are added by `answer`)."""
        pc = self.cfg.pipeline
        passed: list[str] = []
        flags: list[str] = [] if m.hr_times else ["hr_missing"]
        base = {"flags": flags, "model_versions": self.model_versions}
        g = resample_window(m.glucose_times, m.glucose_mgdl, m.meal_time, pc.cgm_lookback_min)
        h = resample_window(m.hr_times, m.hr_bpm, m.meal_time, pc.cgm_lookback_min)
        problem = window_problem(g, self.cfg)
        if problem:
            return cant_judge(problem, checks_failed=["cgm_window"], **base)
        passed.append("cgm_window")
        est = self.photo_estimate(m, key)
        if isinstance(est, tuple):
            check, reason, extra = est
            if m.user_carbs_g is not None:  # tier 2: the user's own carb estimate stands in for the photo
                flags.append("carbs_from_user")
                est = PhotoEstimate(grams=m.user_carbs_g, bin_probs=_bin_onehot(m.user_carbs_g))
                carb_source = "user"
            else:
                return cant_judge(reason, checks_passed=passed, checks_failed=[check],
                                  escalate_to="confirm_carbs", **base, **extra)  # fmt: skip
        else:
            carb_source = "photo"
            passed += ["photo_present", "photo_readable", "photo_not_blank", "photo_familiar"]
        probs = np.array(est.bin_probs)
        carb = {"carbs_g": est.grams, "carb_bin": CARB_BINS[int(probs.argmax())], "carb_bin_prob": float(probs.max()),
                "unfamiliar_photo_score": est.unfamiliar_score, "carb_source": carb_source}  # fmt: skip
        ns = SimpleNamespace(glucose=g.values[None], heart_rate=h.values[None],
                             profile=profile_vector(m.profile, m.meal_time)[None])  # fmt: skip
        x = np.hstack([trend_features(ns, np.array([0])), [[np.log1p(est.grams), *probs[1:]]]])
        risk = float(self.fusion.predict_proba(x)[0, 1])
        dec = decision(risk, self.lo, self.hi)
        if dec == "can't judge":
            return cant_judge(f"the risk ({risk:.0%}) is borderline, between {self.lo:.0%} and {self.hi:.0%}",
                              risk=risk, checks_passed=passed, checks_failed=["borderline"],
                              escalate_to="human", **base, **carb)  # fmt: skip
        passed.append("borderline")
        reason = (f"risk {risk:.0%} is below the low-risk cut-off {self.lo:.0%}" if dec == "low risk"
                  else f"risk {risk:.0%} is at or above the caution cut-off {self.hi:.0%}")  # fmt: skip
        return Answer(decision=dec, sentence=sentence(dec, risk, reason), reason=reason, risk=risk,
                      checks_passed=passed, **base, **carb)  # fmt: skip
