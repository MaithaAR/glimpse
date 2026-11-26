"""Stress quality at the evaluated operating point (`glimpse stress-quality`; registered 26 Nov in
docs/stress_quality.md before its run). Every development meal is answered by its fold's pipeline with
the person's gated history, clean and under each perturbation of the current meal; history always comes
from the clean answers. Both products: v2 (served, sensor first) and v1 (photo fusion)."""

from __future__ import annotations

import json
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

from glimpse.calibration import LABEL_LAG_MIN, choose_m_and_cutoffs
from glimpse.data.splits import meal_rows
from glimpse.evaluation.metrics import apply_platt, fit_platt, usefulness_report
from glimpse.evaluation.run import ACTION, fold_estimates, fold_fusion, inner_oof
from glimpse.serve import Pipeline
from glimpse.studies.robustness import perturb_cgm, perturb_photo
from glimpse.training.final import cost_dict
from glimpse.training.fusion import photo_table

OUT = Path("outputs/robustness/stress_quality.json")
LEVELS = {"cgm_dropout": (0.1, 0.3, 0.5, 0.7), "cgm_gap": (10, 15, 18, 20, 22, 25, 40),
          "photo_blur": (2, 6, 10), "photo_dark": (0.5, 0.25, 0.2, 0.15, 0.1)}  # fmt: skip


def _fold_pipelines(ds, sp, cfg, k: int, sensor_first: bool, oof, nested):
    """The evaluation's fold pipeline (same recipe as evaluate_dev), plus a live-photo twin for v1."""
    y = ds.labels()
    kk = k + 1
    of, nf = oof[oof.fold == kk], nested[nested.fold == kk]
    pt = photo_table(nf) | photo_table(of)
    lr, c = fold_fusion(ds, sp, cfg, k, pt, sensor_first)
    rows_i, p_i = inner_oof(ds, sp, cfg, k, pt, c, sensor_first)
    a, b = fit_platt(p_i, y[rows_i])
    if a <= 0:
        a, b = 1.0, 0.0
    times = np.array([ds.samples[i].meal_time for i in rows_i], dtype="datetime64[s]")
    person = np.array([ds.samples[i].person_id for i in rows_i])
    sel = choose_m_and_cutoffs(apply_platt(p_i, a, b), y[rows_i], person, times, cost_dict(cfg))
    kw = {"fusion": lr, "calibration": (a, b), "m_personal": sel["m"],
          "cutoffs": (sel["low_risk_below"], sel["caution_from"]), "sensor_first": sensor_first}  # fmt: skip
    return Pipeline(cfg, photo_table=fold_estimates(of), **kw), Pipeline(cfg, **kw)


def _summary(clean: pd.DataFrame, pert: pd.DataFrame, costs: dict, targets: dict) -> dict:
    y = clean.spike.to_numpy().astype(int)
    both = clean.risk.notna().to_numpy() & pert.risk.notna().to_numpy()
    b, p = clean.decision.to_numpy(), pert.decision.to_numpy()
    rep = usefulness_report(pert.decision.map(ACTION).to_numpy(), y, costs, targets)
    auc_c = float(roc_auc_score(y[both], clean.risk.to_numpy()[both])) if both.sum() > 10 else None
    auc_p = float(roc_auc_score(y[both], pert.risk.to_numpy()[both])) if both.sum() > 10 else None
    return {"answered_clean": int(clean.risk.notna().sum()), "answered_perturbed": int(pert.risk.notna().sum()),
            "co_answered": int(both.sum()), "auroc_clean": auc_c, "auroc_perturbed": auc_p,
            "auroc_change": None if auc_c is None else auc_p - auc_c,
            "missed_among_cleared": rep["missed_among_cleared"], "cleared": int((p == "low risk").sum()),
            "cost_reduction": rep["cost_reduction"],
            "caution_to_low_risk": int(((b == "caution") & (p == "low risk")).sum()),
            "low_risk_to_caution": int(((b == "low risk") & (p == "caution")).sum()),
            "newly_refused": int(((b != "can't judge") & (p == "can't judge")).sum()),
            "recovered": int(((b == "can't judge") & (p != "can't judge")).sum()), "meals": int(len(b))}  # fmt: skip


def run_stress_quality(ds, sp, cfg) -> dict:
    from glimpse.demo_cases import meal_input

    log, pcv, y = logger.bind(stage="stress-quality"), Path(cfg.paths.outputs) / "photo_cv", ds.labels()
    oof, nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    costs, targets, lag = cost_dict(cfg), cfg.targets.model_dump(), timedelta(minutes=LABEL_LAG_MIN)
    res: dict = {"what": __doc__.splitlines()[0], "levels": LEVELS}
    for product, sensor_first in (("v2_sensor_first", True), ("v1_photo_fusion", False)):
        rng = np.random.default_rng(cfg.seed)
        rows: dict[tuple, list] = {("clean", 0): []} | {(k, lv): [] for k, lvs in LEVELS.items() for lv in lvs}
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for k, held in enumerate(sp.dev_folds):
                tab, live = _fold_pipelines(ds, sp, cfg, k, sensor_first, oof, nested)
                for pid in sorted(held):
                    hist: list = []
                    for row in sorted(meal_rows(ds, {pid}, scored_only=True), key=lambda r: ds.samples[r].meal_time):
                        t = ds.samples[row].meal_time
                        gated = [(z, lab) for z, lab, tm in hist if tm + lag <= t] or None
                        m = meal_input(cfg, ds, int(row))
                        clean = tab.answer(m, key=int(row), history=gated)
                        meta = {"row": int(row), "person_id": int(pid), "spike": int(y[row])}
                        rows[("clean", 0)].append(meta | {"risk": clean.risk, "decision": clean.decision})
                        for kind in ("cgm_dropout", "cgm_gap"):
                            for lv in LEVELS[kind]:
                                a = tab.answer(perturb_cgm(m, kind, rng, lv), key=int(row), history=gated)
                                rows[(kind, lv)].append(meta | {"risk": a.risk, "decision": a.decision})
                        for kind in ("photo_blur", "photo_dark"):
                            for lv in LEVELS[kind]:
                                if m.photo is None:  # no photo to perturb: the clean answer stands
                                    a = clean
                                else:
                                    a = live.answer(perturb_photo(m, kind, tmp, lv), history=gated)
                                rows[(kind, lv)].append(meta | {"risk": a.risk, "decision": a.decision})
                        if clean.fusion_logit is not None:
                            hist.append((clean.fusion_logit, int(y[row]), t))
                log.info("{} fold {} done", product, k + 1)
        base = pd.DataFrame(rows[("clean", 0)])
        out = {"clean": _summary(base, base, costs, targets)}
        for kind, lvs in LEVELS.items():
            out[kind] = {str(lv): _summary(base, pd.DataFrame(rows[(kind, lv)]), costs, targets) for lv in lvs}
        res[product] = out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    for product in ("v2_sensor_first", "v1_photo_fusion"):
        for kind in LEVELS:
            for lv, r in res[product][kind].items():
                log.info("{} {:<11} {:>5}: AUROC {} -> {} | caution->low {} | refused +{}", product, kind, lv,
                         r["auroc_clean"] and round(r["auroc_clean"], 3),
                         r["auroc_perturbed"] and round(r["auroc_perturbed"], 3),
                         r["caution_to_low_risk"], r["newly_refused"])  # fmt: skip
    return res
