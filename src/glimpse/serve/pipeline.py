"""The one scoring path (design decision D2): rules -> carb estimate -> fusion -> cut-offs -> Answer.

`just demo`, `just predict`, the app and the folder command answer through `Pipeline` with the live photo model;
evaluation joins via a precomputed per-fold estimate table through the same decision code. Tier 2 (confirm-carbs,
D5): `MealInput.user_carbs_g` would stand in for an unusable photo, but no entry point sets it yet and its
risk is not monotone in the typed carbs (tests/test_known_failures.py). Report: system diagram, Figure 1 (§1);
the parity test in tests/test_serve.py is the evaluation-equals-deployment proof."""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from loguru import logger

from glimpse.calibration import personal_offset
from glimpse.config import GlimpseConfig
from glimpse.data.photos import PhotoError, eval_transform, open_photo
from glimpse.data.preprocess import resample_window
from glimpse.serve.rules import (
    LOW_GLUCOSE_REFUSE,
    LOW_GLUCOSE_WARN,
    cant_judge,
    decision,
    decision_reason,
    knn_distance,
    low_glucose,
    no_basis_to_clear,
    out_of_scope,
    photo_problem,
    profile_vector,
    sentence,
    window_problem,
)
from glimpse.serve.schemas import CARB_BINS, REQUIRED_PROFILE, Answer, MealInput, PhotoEstimate
from glimpse.training.baselines import trend_features
from glimpse.training.final import FINAL, _photo_checkpoint, load_thresholds
from glimpse.training.photo import load_photo_bundle


def _bin_onehot(grams: float) -> list[float]:
    """The carb range a gram amount falls in, as one-hot probabilities over the four ranges."""
    idx = 0 if grams < 30 else 1 if grams < 60 else 2 if grams < 90 else 3
    return [1.0 if i == idx else 0.0 for i in range(4)]


def heads_disagree(grams: float, probs: np.ndarray) -> bool:
    """The grams head lands outside the range the range head shows (27 Nov; e.g. 36.5 g shown as under 30 g)."""
    return int(np.digitize(grams, [30, 60, 90])) != int(np.argmax(probs))


def _interval_hint(m: MealInput) -> str:
    """When the readings arrive less often than every 5 minutes, say so: the usual cause of a refused window."""
    t = sorted(m.glucose_times)
    if len(t) < 3:
        return ""
    step = float(np.median(np.diff(np.array(t, dtype="datetime64[s]")).astype(float))) / 60
    return (
        f" (readings arrive every {step:.0f} min; GLIMPSE needs a sensor that reads every 5 min)" if step > 7.5 else ""
    )


class Pipeline:
    """Loads the frozen serving pipeline once (fusion fitted on the 37 non-locked people); `answer` scores one meal.

    `photo_table` switches the carb-estimate source from the live photo model to precomputed per-fold estimates
    (evaluation); everything from the rules to the Answer is the same code either way."""

    def __init__(self, cfg: GlimpseConfig, photo_table: dict[int, PhotoEstimate] | None = None, fusion=None,
                 *, calibration: tuple[float, float] | None = None, m_personal: float = float("inf"),
                 cutoffs: tuple[float, float] | None = None, sensor_first: bool | None = None):  # fmt: skip
        """Load the fusion model, the frozen cut-offs (checked against the config) and, in live mode, the photo
        model and the unfamiliar-photo reference, all from local files. Evaluation injects a fold-local fusion
        (design D2: the same decision code, one model per outer fold); the app always loads the final one."""
        torch.set_num_threads(min(cfg.training.num_threads, 4))
        self.cfg, pc = cfg, cfg.prototype
        # v2 (docs/v2_sensor_first.md): risk from trend + profile only; the photo is the carb assist
        self.sensor_first = (pc.product == "sensor_first") if sensor_first is None else sensor_first
        if fusion is None and self.sensor_first:
            bundle = joblib.load(pc.sensor_first_bundle)
            self.fusion = bundle["models"]["sensor"]
            calibration = tuple(bundle["calibration"]) if calibration is None else calibration
            m_personal = float(bundle["m_personal"]) if m_personal == float("inf") else m_personal
        elif fusion is None:  # serving: the frozen bundle carries the fusion AND its calibration (D3/D4 final)
            bundle = joblib.load(FINAL)
            self.fusion = bundle["models"]["photo"]
            if calibration is None and "calibration" in bundle:
                calibration = tuple(bundle["calibration"])
            if m_personal == float("inf") and "m_personal" in bundle:
                m_personal = float(bundle["m_personal"])
        else:
            self.fusion = fusion
        if self.sensor_first and fusion is None:
            frozen_all = load_thresholds(Path(pc.sensor_first_thresholds))
            frozen = frozen_all["main"]
            served = (frozen["low_risk_below"], frozen["caution_from"])
        else:
            frozen_all = load_thresholds(Path(pc.thresholds_file))
            frozen = frozen_all["main"]
            if max(abs(frozen["low_risk_below"] - pc.low_risk_below),
                   abs(frozen["caution_from"] - pc.caution_from)) > 1e-9:  # fmt: skip
                raise ValueError("prototype cut-offs in the config differ from the frozen amendment file")
            served = (pc.low_risk_below, pc.caution_from)
        # D4: evaluation injects each fold's inner-chosen cut-offs; the app uses the frozen values
        self.lo, self.hi = cutoffs if cutoffs is not None else served
        # fix 1 (D3): Platt (slope, intercept) on the fusion logit + the personal-intercept prior precision.
        # The defaults are the identity; the frozen values come from `just finalize` (6 Nov) via the bundle.
        self.calib, self.m_personal = calibration, m_personal
        self.table = photo_table
        name = Path(pc.sensor_first_bundle).name if self.sensor_first else FINAL.name
        self.model_versions = {"fusion": "fold-local" if fusion is not None else name,
                               "product": "sensor_first" if self.sensor_first else "photo_fusion",
                               "thresholds": str(frozen_all.get("sha256", ""))[:12]}  # fmt: skip
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

    def answer(self, m: MealInput, key: int | None = None,
               history: list[tuple[float, int]] | None = None) -> Answer:  # fmt: skip
        """One meal -> one validated Answer. Rules first; the models only see inputs that passed them.
        history: the person EARLIER meals as (raw fusion logit, label) pairs -- only meals whose CGM
        label was already available at this meal time (the caller enforces the two-hour gate; the app
        gets this for free because labels only exist once the window closes)."""
        t0 = time.perf_counter()
        a = self._answer(m, key, history)
        if a.decision != "can't judge" and m.glucose_mgdl:  # rule R14, 54-69 mg/dL: wording only
            last = m.glucose_mgdl[max(range(len(m.glucose_times)), key=m.glucose_times.__getitem__)]
            if LOW_GLUCOSE_REFUSE <= last < LOW_GLUCOSE_WARN:
                a.sentence = sentence(a.decision, a.risk, a.reason, low_mgdl=last)
                a.flags.append("glucose_low")
        a.latency_ms = (time.perf_counter() - t0) * 1000
        msg = "{} ({:.0f} ms): {}"
        (self.log.info if a.decision == "can't judge" else self.log.debug)(msg, a.decision, a.latency_ms, a.reason)
        return a

    def _answer_sensor_first(self, m: MealInput, est, g, h, history, passed: list[str], flags: list[str],
                             base: dict) -> Answer:  # fmt: skip
        """v2 (docs/v2_sensor_first.md): risk from the 15 trend + profile inputs; the photo estimate, when
        usable, is returned as the carb assist for the user to confirm and never enters the risk."""
        if isinstance(est, tuple):
            check, _reason, extra = est
            flags.append(f"carbs_unavailable_{check}")
            carb = {"carb_source": "none", **{k: v for k, v in extra.items() if k == "unfamiliar_photo_score"}}
        else:
            probs = np.array(est.bin_probs)
            passed += ["photo_present", "photo_readable", "photo_not_blank", "photo_familiar"]
            carb = {"carbs_g": est.grams, "carb_bin": CARB_BINS[int(probs.argmax())],
                    "carb_bin_prob": float(probs.max()), "unfamiliar_photo_score": est.unfamiliar_score,
                    "carb_source": "photo"}  # fmt: skip
            if heads_disagree(est.grams, probs):
                flags.append("carb_heads_disagree")
        if m.user_carbs_g is not None:
            flags.append("carbs_from_user_shown_not_used")
        ns = SimpleNamespace(glucose=g.values[None], heart_rate=h.values[None],
                             profile=profile_vector(m.profile, m.meal_time)[None])  # fmt: skip
        raw = float(self.fusion.predict_proba(trend_features(ns, np.array([0])))[0, 1])
        risk_raw, risk_platt, risk = self._risk(raw, history)
        base |= {"fusion_logit": float(np.log(max(raw, 1e-6) / max(1.0 - raw, 1e-6))),
                 "risk_raw": risk_raw, "risk_platt": risk_platt}  # fmt: skip
        dec = decision(risk, self.lo, self.hi)
        if dec == "can't judge":
            return cant_judge(f"the risk ({risk:.1%}) is borderline, between {self.lo:.0%} and {self.hi:.0%}",
                              risk=risk, checks_passed=passed, checks_failed=["borderline"],
                              escalate_to="human", **base, **carb)  # fmt: skip
        passed.append("borderline")
        hold = no_basis_to_clear(dec, history, m.profile)
        if hold:  # rule R13
            return cant_judge(hold, risk=risk, checks_passed=passed, checks_failed=["personal_basis"],
                              escalate_to="human", **base, **carb)  # fmt: skip
        reason = decision_reason(dec, risk, self.lo, self.hi)
        return Answer(decision=dec, sentence=sentence(dec, risk, reason), reason=reason, risk=risk,
                      checks_passed=passed, **base, **carb)  # fmt: skip

    def _risk(self, raw: float, history: list[tuple[float, int]] | None) -> tuple[float, float, float]:
        """raw fusion probability -> (risk_raw, risk_platt, final risk): Platt on the logit, then the
        personal offset fitted on the (equally recalibrated) history logits (D3). Identity by default."""
        if self.calib is None and not history:
            return raw, raw, raw  # the identity path is literally the identity (no logit round-trip)
        z = float(np.log(max(raw, 1e-6) / max(1.0 - raw, 1e-6)))
        f = self.calib[0] * z + self.calib[1] if self.calib is not None else z
        platt = raw if self.calib is None else 1.0 / (1.0 + float(np.exp(-f)))
        if history:
            hz = np.array([h[0] for h in history], dtype=float)
            hf = self.calib[0] * hz + self.calib[1] if self.calib is not None else hz
            f += personal_offset(hf, np.array([h[1] for h in history], dtype=float), self.m_personal)
        return raw, platt, float(1.0 / (1.0 + np.exp(-f)))

    def _answer(self, m: MealInput, key: int | None = None,
                history: list[tuple[float, int]] | None = None) -> Answer:  # fmt: skip
        """The decision itself (timing and logging are added by `answer`)."""
        pc = self.cfg.pipeline
        passed: list[str] = []
        flags: list[str] = list(m.input_flags) + ([] if m.hr_times else ["hr_missing"])
        base = {"flags": flags, "model_versions": self.model_versions}
        if m.glucose_mgdl and min(m.glucose_mgdl) <= 0:  # rule R12: zero or negative is a broken export, not mmol/L
            return cant_judge("a glucose reading is zero or negative; the export is broken, check the file",
                              checks_failed=["cgm_range"], **base)  # fmt: skip
        if m.glucose_mgdl and max(m.glucose_mgdl) < 30:  # rule R1: a mmol/L export reads as 4-9, never as mg/dL
            return cant_judge("the glucose values look like mmol/L; multiply by 18 or export the file in mg/dL",
                              checks_failed=["cgm_units"], **base)  # fmt: skip
        passed.append("cgm_units")
        late = [t for t in m.glucose_times if t > m.meal_time]
        if late:  # rule R3: a sensor cannot report the future; the clock or the export is wrong
            return cant_judge(f"{len(late)} glucose readings are timestamped after the meal time "
                              f"(latest {max(late):%Y-%m-%d %H:%M}); check the export and the meal time",
                              checks_passed=passed.copy(), checks_failed=["cgm_times"], **base)  # fmt: skip
        passed.append("cgm_times")
        g = resample_window(m.glucose_times, m.glucose_mgdl, m.meal_time, pc.cgm_lookback_min)
        h = resample_window(m.hr_times, m.hr_bpm, m.meal_time, pc.cgm_lookback_min)
        problem = window_problem(g, self.cfg)
        if problem:
            return cant_judge(problem + _interval_hint(m), checks_failed=["cgm_window"], **base)
        passed.append("cgm_window")
        last = m.glucose_mgdl[max(range(len(m.glucose_times)), key=m.glucose_times.__getitem__)]
        low = low_glucose(last)
        if low:  # rule R14: a low now needs treating, not a forecast
            return cant_judge(low, checks_passed=passed.copy(), checks_failed=["glucose_low"], escalate_to="human",
                              **base)  # fmt: skip
        passed.append("glucose_not_low")
        blank = [k for k in REQUIRED_PROFILE if not math.isfinite(m.profile.get(k, float("nan")))]
        if blank:  # rule R5: a missing profile is named, never silently filled with defaults
            return cant_judge("profile fields missing or blank: " + ", ".join(blank),
                              checks_passed=passed.copy(), checks_failed=["profile"], **base)  # fmt: skip
        passed.append("profile")
        scope = out_of_scope(m.profile)
        if scope:  # rule R15: no claim is made for the diabetes range
            return cant_judge(scope, checks_passed=passed.copy(), checks_failed=["in_scope"], escalate_to="human",
                              **base)  # fmt: skip
        passed.append("in_scope")
        est = self.photo_estimate(m, key)
        if getattr(self, "sensor_first", False):  # v2: the photo never blocks the answer; it only feeds the carb assist
            return self._answer_sensor_first(m, est, g, h, history, passed, flags, base)
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
        if carb_source == "photo" and heads_disagree(est.grams, probs):
            flags.append("carb_heads_disagree")
        carb = {"carbs_g": est.grams, "carb_bin": CARB_BINS[int(probs.argmax())], "carb_bin_prob": float(probs.max()),
                "unfamiliar_photo_score": est.unfamiliar_score, "carb_source": carb_source}  # fmt: skip
        ns = SimpleNamespace(glucose=g.values[None], heart_rate=h.values[None],
                             profile=profile_vector(m.profile, m.meal_time)[None])  # fmt: skip
        x = np.hstack([trend_features(ns, np.array([0])), [[np.log1p(est.grams), *probs[1:]]]])
        raw = float(self.fusion.predict_proba(x)[0, 1])
        risk_raw, risk_platt, risk = self._risk(raw, history)
        base |= {"fusion_logit": float(np.log(max(raw, 1e-6) / max(1.0 - raw, 1e-6))),
                 "risk_raw": risk_raw, "risk_platt": risk_platt}  # fmt: skip
        dec = decision(risk, self.lo, self.hi)
        if dec == "can't judge":
            return cant_judge(f"the risk ({risk:.1%}) is borderline, between {self.lo:.0%} and {self.hi:.0%}",
                              risk=risk, checks_passed=passed, checks_failed=["borderline"],
                              escalate_to="human", **base, **carb)  # fmt: skip
        passed.append("borderline")
        hold = no_basis_to_clear(dec, history, m.profile)
        if hold:  # rule R13: no low risk on the population calibration alone
            return cant_judge(hold, risk=risk, checks_passed=passed, checks_failed=["personal_basis"],
                              escalate_to="human", **base, **carb)  # fmt: skip
        reason = decision_reason(dec, risk, self.lo, self.hi)
        return Answer(decision=dec, sentence=sentence(dec, risk, reason), reason=reason, risk=risk,
                      checks_passed=passed, **base, **carb)  # fmt: skip
