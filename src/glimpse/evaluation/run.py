"""Evaluation v2 on partition v2 (designs D1-D4): every reported development risk and decision is
produced by `Pipeline.answer`, one Pipeline per outer fold carrying that fold's fusion, its Platt map
(fitted on inner-fold out-of-fold risk), its prior precision m and its cut-offs (m, lo, hi chosen
together by minimum expected cost on the replayed inner risks -- fix 2). Held-out meals are answered in
deployment order: the personal intercept (fix 1) sees only earlier meals whose two-hour CGM label window
has closed, fed back as the Pipeline's own fusion logits. A meal without a photo is the same 'can't
judge' the app gives. The sensitivity sweep re-chooses cut-offs per pre-registered cost setting with the
main m fixed. Report: §3 evaluation, §4 trust."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss, roc_auc_score

from glimpse.calibration import LABEL_LAG_MIN, choose_m_and_cutoffs, sweep_cutoffs
from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import SplitsV2, assert_not_locked, meal_rows
from glimpse.evaluation.metrics import (
    apply_platt,
    calibration_by_person,
    fit_platt,
    person_bootstrap_ci,
    usefulness_report,
)
from glimpse.serve import PhotoEstimate, Pipeline
from glimpse.training.baselines import logistic
from glimpse.training.cv import _tune_sklearn
from glimpse.training.final import cost_dict
from glimpse.training.fusion import PHOTO_COLS, photo_table, with_table

OUT = Path("outputs/evaluation_v2")
ACTION = {"low risk": 0, "caution": 1, "can't judge": 2}


def fold_estimates(df: pd.DataFrame) -> dict[int, PhotoEstimate]:
    """Out-of-fold photo CV rows -> the Pipeline's precomputed-estimate table for the held-out people."""
    return {int(r.row): PhotoEstimate(grams=float(r.photo_g),
                                      bin_probs=[float(r.p_bin0), float(r.p_bin1), float(r.p_bin2),
                                                 float(r.p_bin3)])
            for r in df.itertuples()}  # fmt: skip


def fold_fusion(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig, k: int, pt: dict) -> tuple:
    """Fold k's fusion: C chosen on the fold's inner folds (target holds, D1), fitted on its training meals
    with nested photo estimates -- features in exactly the order Pipeline.answer builds them."""
    y = ds.labels()
    fx = with_table(pt, len(PHOTO_COLS))
    c, _ = _tune_sklearn(lambda v: logistic(v, cfg.seed), cfg.baselines.logreg_c_grid, fx, ds,
                         sp.training_people(k), sp.inner_folds[k], y)  # fmt: skip
    tr = meal_rows(ds, set(sp.training_people(k)), scored_only=False)
    return logistic(c, cfg.seed).fit(fx(ds, tr), y[tr]), c


def inner_oof(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig, k: int, pt: dict, c: float) -> tuple:
    """Inner-fold out-of-fold fusion risks for fold k's training targets: the fold's recipe (same C)
    refitted without each inner fold, used on its held-out target meals. These risks feed Platt and the
    (m, lo, hi) selection; they never reach a results table."""
    y, fx = ds.labels(), with_table(pt, len(PHOTO_COLS))
    rows, risks = [], []
    for held in sp.inner_folds[k]:
        fit = meal_rows(ds, set(sp.training_people(k)) - set(held), scored_only=False)
        val = meal_rows(ds, set(held), scored_only=True)
        m = logistic(c, cfg.seed).fit(fx(ds, fit), y[fit])
        rows.append(val)
        risks.append(m.predict_proba(fx(ds, val))[:, 1])
    rows, risks = np.concatenate(rows), np.concatenate(risks)
    targets = {p for f in sp.inner_folds[k] for p in f}
    expect = meal_rows(ds, targets, scored_only=True)
    if sorted(rows.tolist()) != sorted(expect.tolist()):
        raise RuntimeError(f"fold {k + 1}: inner-OOF coverage is not exactly once per training-target meal")
    return rows, risks


def _sweep_costs(cfg: GlimpseConfig) -> list[dict]:
    """The pre-registered cost settings for the sensitivity check: missed-spike cost x abstain multiplier."""
    c = cfg.costs
    return [{"missed_spike": m, "abstain_multiplier": k, "costs": {"low_risk": (m, 0.0), "caution": tuple(c.caution),
             "abstain": (c.abstain[0] * k, c.abstain[1] * k)}}
            for m in c.sweep_missed_spike for k in c.sweep_abstain_multiplier]  # fmt: skip


def _replay_fold(app: Pipeline, ds: MealDataset, cfg: GlimpseConfig, held, y, kk: int) -> list[dict]:
    """Answer fold kk's held-out meals in deployment order: each person's history holds only earlier meals
    whose label window closed (two hours) and which produced a fusion logit (D3 as amended)."""
    from glimpse.demo_cases import meal_input

    lag, out = timedelta(minutes=LABEL_LAG_MIN), []
    for pid in sorted(held):
        hist: list[tuple[float, int, object]] = []
        for row in sorted(meal_rows(ds, {pid}, scored_only=True), key=lambda r: ds.samples[r].meal_time):
            t = ds.samples[row].meal_time
            gated = [(z, lab) for z, lab, tm in hist if tm + lag <= t]
            a = app.answer(meal_input(cfg, ds, int(row)), key=int(row), history=gated or None)
            out.append({"fold": kk, "row": int(row), "person_id": int(pid), "spike": int(y[row]),
                        "risk": a.risk, "risk_platt": a.risk_platt, "risk_raw": a.risk_raw,
                        "n_history": len(gated), "decision": a.decision, "carb_source": a.carb_source,
                        "reason": a.reason})  # fmt: skip
            if a.fusion_logit is not None:
                hist.append((a.fusion_logit, int(y[row]), t))
    return out


def _risk_metrics(df: pd.DataFrame, col: str, B: int, seed: int) -> dict:
    """AUROC (person-clustered bootstrap) and log-loss of one risk column over the answered meals."""
    d = df.dropna(subset=[col])
    s, y, p = d.person_id.to_numpy(), d.spike.to_numpy(), d[col].to_numpy()
    auc = person_bootstrap_ci(s, lambda i: roc_auc_score(y[i], p[i]), B, seed)
    return {"auroc": {"point": auc[0], "ci95": [auc[1], auc[2]]},
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "n": int(len(d))}  # fmt: skip


def _fold_auroc_table(df: pd.DataFrame) -> list[dict]:
    """Per-fold AUROC of the three risk columns. Platt is a monotone map fitted per fold, so within every
    fold auroc_platt == auroc_raw; only pooling across folds can reorder meals (the report's footnote)."""
    out = []
    for kk, g in df.dropna(subset=["risk"]).groupby("fold"):
        yk = g.spike.to_numpy()
        row = {"fold": int(kk), "n": int(len(g))}
        for name, col in (("raw", "risk_raw"), ("platt", "risk_platt"), ("personal", "risk")):
            row[f"auroc_{name}"] = float(roc_auc_score(yk, g[col].to_numpy()))
        out.append(row)
    return out


def evaluate_dev(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    """Every development meal answered by its fold's Pipeline -> dev_answers.csv + dev_summary.json."""
    log, pcv = logger.bind(stage="evaluate"), Path(cfg.paths.outputs) / "photo_cv"
    if not (pcv / "photo_oof.csv").is_file():
        raise FileNotFoundError("run `just photo-cv` first")
    oof, nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    y, costs, rows_out, folds, inner_store = ds.labels(), cost_dict(cfg), [], [], {}
    for k, held in enumerate(sp.dev_folds):
        kk = k + 1
        of, nf = oof[oof.fold == kk], nested[nested.fold == kk]
        if of.empty or not set(of.person_id) <= set(held):  # provenance: must be the partition-v2 run
            raise RuntimeError(f"fold {kk}: photo CV outputs do not match splits/dev_splits_v2.json; "
                               f"rerun `just photo-cv`")  # fmt: skip
        assert_not_locked(list(sp.training_people(k)) + list(held), sp.locked, f"evaluate fold {kk}")
        pt = photo_table(nf) | photo_table(of)
        lr, c = fold_fusion(ds, sp, cfg, k, pt)
        rows_i, p_i = inner_oof(ds, sp, cfg, k, pt, c)
        a_k, b_k = fit_platt(p_i, y[rows_i])
        if a_k <= 0:  # degenerate inner risks would invert the map; fall back to identity, loudly
            log.warning("fold {}: Platt slope {:.3f} <= 0; identity used instead", kk, a_k)
            a_k, b_k = 1.0, 0.0
        times_i = np.array([ds.samples[i].meal_time for i in rows_i], dtype="datetime64[s]")
        person_i = np.array([ds.samples[i].person_id for i in rows_i])
        sel = choose_m_and_cutoffs(apply_platt(p_i, a_k, b_k), y[rows_i], person_i, times_i, costs)
        app = Pipeline(cfg, photo_table=fold_estimates(of), fusion=lr, calibration=(a_k, b_k),
                       m_personal=sel["m"], cutoffs=(sel["low_risk_below"], sel["caution_from"]))  # fmt: skip
        rows_out += _replay_fold(app, ds, cfg, held, y, kk)
        inner_store[kk] = (sel.pop("inner_risks"), y[rows_i])
        folds.append({"fold": kk, "C": c, "platt_slope": a_k, "platt_intercept": b_k, "m": sel["m"],
                      "low_risk_below": sel["low_risk_below"], "caution_from": sel["caution_from"],
                      "inner_cost": sel["cost"], "cost_vs_m_curve": sel["curve"]})  # fmt: skip
        log.info("fold {}: C={}, Platt ({:.2f}, {:+.2f}), m={}, cut-offs ({:.2f}, {:.2f})", kk, c, a_k, b_k,
                 sel["m"], sel["low_risk_below"], sel["caution_from"])  # fmt: skip
    df = pd.DataFrame(rows_out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "dev_answers.csv", index=False)
    B, seed, targets = cfg.fusion.bootstrap_draws, cfg.seed, cfg.targets.model_dump()
    actions = df.decision.map(ACTION).to_numpy()
    answered = df.dropna(subset=["risk"])
    res = {
        "note": "every risk and decision from Pipeline.answer, one Pipeline per fold; m and cut-offs chosen "
                "inside each fold's inner folds only (D4 realised per fold; deployment values frozen later)",
        "meals": int(len(df)), "answered_with_risk": int(len(answered)),
        "risk_personal": _risk_metrics(df, "risk", B, seed),
        "risk_platt_only": _risk_metrics(df, "risk_platt", B, seed),
        "risk_raw": _risk_metrics(df, "risk_raw", B, seed),
        "per_fold_auroc": _fold_auroc_table(df),
        "risk_personal_excluding_fold1": _risk_metrics(df[df.fold != 1], "risk", B, seed),
        "decisions": usefulness_report(actions, df.spike.to_numpy(), costs, targets),
        "abstained_no_photo": int((df.decision == "can't judge").sum() - (df.risk.notna()
                                  & (df.decision == "can't judge")).sum()),  # fmt: skip
        "abstained_borderline": int((df.risk.notna() & (df.decision == "can't judge")).sum()),
        "calibration_by_person_personal": calibration_by_person(answered.person_id.to_numpy(),
                                                                answered.spike.to_numpy(),
                                                                answered.risk.to_numpy()),  # fmt: skip
        "calibration_by_person_platt_only": calibration_by_person(answered.person_id.to_numpy(),
                                                                  answered.spike.to_numpy(),
                                                                  answered.risk_platt.to_numpy()),  # fmt: skip
        "folds": folds,
        "cost_sweep": [],
    }  # fmt: skip
    for scn in _sweep_costs(cfg):
        acts = np.full(len(df), 2)
        cuts = {}
        for fr in folds:
            kk = fr["fold"]
            lo, hi = sweep_cutoffs(*inner_store[kk], scn["costs"])
            cuts[kk] = (lo, hi)
            i = (df.fold == kk).to_numpy()
            r = df.risk.to_numpy()[i]
            # the same rule serve.rules.decision applies, vectorised over the Pipeline's own risks
            acts[i] = np.where(np.isnan(r), 2, np.where(r < lo, 0, np.where(r >= hi, 1, 2)))
        rep = usefulness_report(acts, df.spike.to_numpy(), scn["costs"], targets)
        res["cost_sweep"].append({"missed_spike": scn["missed_spike"],
                                  "abstain_multiplier": scn["abstain_multiplier"],
                                  "cutoffs_per_fold": cuts, "cost_reduction": rep["cost_reduction"],
                                  "share_low_risk": rep["share_low_risk"],
                                  "share_abstain": rep["share_abstain"],
                                  "missed_among_cleared": rep["missed_among_cleared"]})  # fmt: skip
    (OUT / "dev_summary.json").write_text(json.dumps(res, indent=2))
    rp = res["risk_personal"]["auroc"]
    log.info("development: personal AUROC {:.3f} ({:.3f}-{:.3f}) | platt-only {:.3f} | raw {:.3f}; "
             "log-loss {:.3f} / {:.3f} / {:.3f}", rp["point"], *rp["ci95"],
             res["risk_platt_only"]["auroc"]["point"], res["risk_raw"]["auroc"]["point"],
             res["risk_personal"]["log_loss"], res["risk_platt_only"]["log_loss"],
             res["risk_raw"]["log_loss"])  # fmt: skip
    d = res["decisions"]
    log.info("decisions (per-fold cut-offs): low risk {:.0%}, caution {:.0%}, can't judge {:.0%}; "
             "cost reduction {:.0%} vs best trivial; missed among cleared {:.1%}; useful={}",
             d["share_low_risk"], d["share_caution"], d["share_abstain"], d["cost_reduction"],
             d["missed_among_cleared"], d["useful"])  # fmt: skip
    return res
