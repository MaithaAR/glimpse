"""S9 robustness: the four pre-registered input perturbations, each answered by the SAME per-fold
Pipelines the evaluation uses (fold fusion, Platt, cut-offs; no history, so every comparison is at the
population-calibrated operating point and differences come only from the perturbation). A flip into
"can't judge" is the system refusing degraded input -- the designed response -- and is reported
separately from the dangerous flips between low risk and caution. Report: §4 trust, one figure."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from PIL import ImageEnhance, ImageFilter

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.photos import open_photo
from glimpse.data.splits import SplitsV2, meal_rows
from glimpse.evaluation.metrics import apply_platt, fit_platt
from glimpse.evaluation.run import fold_estimates, fold_fusion, inner_oof
from glimpse.serve import MealInput, Pipeline
from glimpse.training.fusion import photo_table

OUT = Path("outputs/robustness")
DROP_SHARE, GAP_MIN, BLUR_RADIUS, DARKEN = 0.3, 25, 6, 0.25


def perturb_cgm(m: MealInput, kind: str, rng: np.random.Generator) -> MealInput:
    """Dropout: each reading lost with p=0.3. Gap: a contiguous 25-minute block removed, ending half an
    hour before the meal."""
    t, v = list(m.glucose_times), list(m.glucose_mgdl)
    if kind == "cgm_dropout":
        keep = rng.uniform(size=len(t)) > DROP_SHARE
        keep[-1] = True  # the sensor's latest reading is always there
        t, v = [x for x, k in zip(t, keep, strict=True) if k], [x for x, k in zip(v, keep, strict=True) if k]
    else:  # cgm_gap
        end = m.meal_time - pd.Timedelta(minutes=30)
        start = end - pd.Timedelta(minutes=GAP_MIN)
        pairs = [(x, y) for x, y in zip(t, v, strict=True) if not (start <= x < end)]
        t, v = [p[0] for p in pairs], [p[1] for p in pairs]
    return m.model_copy(update={"glucose_times": t, "glucose_mgdl": v})


def perturb_photo(m: MealInput, kind: str, tmp: Path) -> MealInput:
    """Blur (radius 6) or darken (x0.25), written next to nothing the user owns."""
    img = open_photo(m.photo)
    img = img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)) if kind == "photo_blur" \
        else ImageEnhance.Brightness(img).enhance(DARKEN)  # fmt: skip
    out = tmp / f"{kind}_{Path(m.photo).name}"
    img.save(out, quality=90)
    return m.model_copy(update={"photo": out})


def _flips(base: list[str], pert: list[str]) -> dict:
    b, p = np.array(base), np.array(pert)
    danger = ((b == "low risk") & (p == "caution")) | ((b == "caution") & (p == "low risk"))
    return {"n": int(len(b)), "unchanged": float((b == p).mean()),
            "dangerous_flips": float(danger.mean()),
            "newly_cant_judge": float(((b != "can't judge") & (p == "can't judge")).mean()),
            "recovered_answers": float(((b == "can't judge") & (p != "can't judge")).mean())}  # fmt: skip


def run_robustness(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    """Four perturbations x every development meal, through the fold's own Pipeline."""
    from glimpse.demo_cases import meal_input

    log, pcv, y = logger.bind(stage="robust"), Path(cfg.paths.outputs) / "photo_cv", ds.labels()
    oof, nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    from glimpse.calibration import choose_m_and_cutoffs
    from glimpse.training.final import cost_dict

    base_tab, base_live = [], []
    pert = {k: [] for k in ("cgm_dropout", "cgm_gap", "photo_blur", "photo_dark")}
    rng = np.random.default_rng(cfg.seed)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for k, held in enumerate(sp.dev_folds):
            kk = k + 1
            of, nf = oof[oof.fold == kk], nested[nested.fold == kk]
            pt = photo_table(nf) | photo_table(of)
            lr, c = fold_fusion(ds, sp, cfg, k, pt)
            rows_i, p_i = inner_oof(ds, sp, cfg, k, pt, c)
            a_k, b_k = fit_platt(p_i, y[rows_i])
            if a_k <= 0:
                a_k, b_k = 1.0, 0.0
            times_i = np.array([ds.samples[i].meal_time for i in rows_i], dtype="datetime64[s]")
            person_i = np.array([ds.samples[i].person_id for i in rows_i])
            sel = choose_m_and_cutoffs(apply_platt(p_i, a_k, b_k), y[rows_i], person_i, times_i, cost_dict(cfg))
            cuts = (sel["low_risk_below"], sel["caution_from"])
            tab = Pipeline(cfg, photo_table=fold_estimates(of), fusion=lr, calibration=(a_k, b_k), cutoffs=cuts)
            live = Pipeline(cfg, fusion=lr, calibration=(a_k, b_k), cutoffs=cuts)
            for row in meal_rows(ds, set(held), scored_only=True):
                m = meal_input(cfg, ds, int(row))
                base_tab.append(tab.answer(m, key=int(row)).decision)
                for kind in ("cgm_dropout", "cgm_gap"):
                    pert[kind].append(tab.answer(perturb_cgm(m, kind, rng), key=int(row)).decision)
                if m.photo is None:
                    base_live.append(None)
                    pert["photo_blur"].append(None)
                    pert["photo_dark"].append(None)
                    continue
                base_live.append(live.answer(m).decision)
                for kind in ("photo_blur", "photo_dark"):
                    pert[kind].append(live.answer(perturb_photo(m, kind, tmp)).decision)
            log.info("fold {} perturbed", kk)
    has_photo = [b is not None for b in base_live]
    res = {"what": "decision-flip rates under the four pre-registered perturbations; same per-fold "
                   "Pipelines as the evaluation, no history (population operating point)",
           "settings": {"dropout_share": DROP_SHARE, "gap_min": GAP_MIN, "blur_radius": BLUR_RADIUS,
                        "darken_factor": DARKEN},
           "cgm_dropout": _flips(base_tab, pert["cgm_dropout"]),
           "cgm_gap": _flips(base_tab, pert["cgm_gap"]),
           "photo_blur": _flips([b for b in base_live if b is not None],
                                [p for p, h in zip(pert["photo_blur"], has_photo, strict=True) if h]),
           "photo_dark": _flips([b for b in base_live if b is not None],
                                [p for p, h in zip(pert["photo_dark"], has_photo, strict=True) if h])}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robustness.json").write_text(json.dumps(res, indent=2))
    for kind in ("cgm_dropout", "cgm_gap", "photo_blur", "photo_dark"):
        r = res[kind]
        log.info("{:<12} unchanged {:.0%} | dangerous flips {:.1%} | newly can't-judge {:.0%} (n={})",
                 kind, r["unchanged"], r["dangerous_flips"], r["newly_cant_judge"], r["n"])  # fmt: skip
    return res
