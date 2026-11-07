"""`glimpse promised` (pre-registered 8 Nov, docs/final_design.md): the comparisons the intermediate
promised (its lines 136-138) that were never run, executed once each against the frozen evidence with
the October bars quoted in the design doc. Descriptive only: nothing shipped changes whatever these
say. The refit photo arm must reproduce the committed out-of-fold predictions exactly, or we stop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import SplitsV2, meal_rows
from glimpse.evaluation.metrics import ece_equal_mass, noninferiority, paired_person_bootstrap
from glimpse.training.baselines import gbm, logistic
from glimpse.training.cv import _tune_sklearn
from glimpse.training.fusion import PHOTO_COLS, photo_table, with_table

OUT = Path("outputs/promised")


def bin_agreement(grams: np.ndarray, probs: np.ndarray) -> float:
    """R5: share of photos where the grams head and the classifier head land in the same carb bin."""
    return float((np.digitize(grams, (30, 60, 90)) == probs.argmax(axis=1)).mean())


def weighted(c: float, seed: int):
    """The class-weighted variant of the shipped fusion recipe (R2); everything else identical."""
    est = logistic(c, seed)
    est[-1].set_params(class_weight="balanced")
    return est


def _paired(y, a, b, person) -> dict:
    point, lo, hi = paired_person_bootstrap(np.asarray(y), np.asarray(a), np.asarray(b), np.asarray(person))
    return {"point": point, "ci95": [lo, hi]}


def run_promised(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    log = logger.bind(stage="promised")
    OUT.mkdir(parents=True, exist_ok=True)
    pcv = Path(cfg.paths.outputs) / "photo_cv"
    p_oof, p_nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    fo = pd.read_csv(Path(cfg.paths.outputs) / "fusion" / "fusion_oof.csv")
    y = ds.labels()
    rows_all, person, y_te, lr_p, gbm_p, wt_p, mask_p, nomeal_p = [], [], [], [], [], [], [], []
    for k, fold in enumerate(sp.dev_folds):
        kk = k + 1
        pt = photo_table(p_nested[p_nested.fold == kk]) | photo_table(p_oof[p_oof.fold == kk])
        fx = with_table(pt, len(PHOTO_COLS))
        people, inner = sp.training_people(k), sp.inner_folds[k]
        tr = meal_rows(ds, set(people), scored_only=False)
        te = [int(i) for i in meal_rows(ds, set(fold), scored_only=True)]
        c, _ = _tune_sklearn(lambda v: logistic(v, cfg.seed), cfg.baselines.logreg_c_grid, fx, ds,
                             people, inner, y)  # fmt: skip
        lr = logistic(c, cfg.seed).fit(fx(ds, tr), y[tr])
        mine = lr.predict_proba(fx(ds, te))[:, 1]
        sub = fo[fo.fold == kk].set_index("row")
        committed = sub.loc[te, "photo"].to_numpy()
        drift = float(np.max(np.abs(mine - committed)))
        if drift > 1e-9:
            raise RuntimeError(f"fold {kk}: refit photo arm drifts from the committed OOF by {drift:.2e}")
        x_te = fx(ds, te)
        x_masked = x_te.copy()
        x_masked[:, -len(PHOTO_COLS):] = np.nan  # R4: photo inputs at their imputation medians
        d, _ = _tune_sklearn(lambda v: gbm(v, cfg.seed), cfg.baselines.gbm_depth_grid, fx, ds,
                             people, inner, y)  # fmt: skip
        gb = gbm(d, cfg.seed).fit(fx(ds, tr), y[tr])
        cw, _ = _tune_sklearn(lambda v: weighted(v, cfg.seed), cfg.baselines.logreg_c_grid, fx, ds,
                              people, inner, y)  # fmt: skip
        wt = weighted(cw, cfg.seed).fit(fx(ds, tr), y[tr])
        rows_all += te
        person += [ds.samples[i].person_id for i in te]
        y_te += [int(y[i]) for i in te]
        lr_p += list(committed)
        gbm_p += list(gb.predict_proba(x_te)[:, 1])
        wt_p += list(wt.predict_proba(x_te)[:, 1])
        mask_p += list(lr.predict_proba(x_masked)[:, 1])
        nomeal_p += list(sub.loc[te, "no_meal"].to_numpy())
        log.info("fold {}: C={} (committed arm reproduced, drift {:.1e}), gbm depth={}, weighted C={}",
                 kk, c, drift, d, cw)  # fmt: skip
    yv, pv = np.array(y_te), np.array(person)
    r1 = _paired(yv, lr_p, gbm_p, pv)
    r1 |= {"auroc_lr": float(roc_auc_score(yv, lr_p)), "auroc_gbm": float(roc_auc_score(yv, gbm_p)),
           "bar": "shipped LR non-inferior to GBM if paired lower bound > -0.03 (intermediate line 137)",
           "verdict": noninferiority(r1["ci95"][0], r1["ci95"][1])}  # fmt: skip
    r2 = _paired(yv, wt_p, lr_p, pv)
    r2 |= {"auroc_weighted": float(roc_auc_score(yv, wt_p)),
           "bar": "adopt only if lower bound > 0 (adoption impossible under the freeze; verdict descriptive)",
           "verdict": "would adopt" if r2["ci95"][0] > 0 else "no gain established"}  # fmt: skip
    r4 = _paired(yv, mask_p, nomeal_p, pv)
    r4 |= {"auroc_masked": float(roc_auc_score(yv, mask_p)),
           "auroc_no_meal": float(roc_auc_score(yv, nomeal_p)),
           "note": "masked photo arm vs the no-meal arm; expected near zero (design doc, 8 Nov)"}  # fmt: skip
    da = pd.read_csv(Path(cfg.paths.outputs) / "evaluation_v2" / "dev_answers.csv")
    da = da[da.risk.notna()]
    rise = np.array([ds.samples[int(r)].rise_mgdl for r in da.row])
    r3 = {f"spike_{t}": {"auroc_personal": float(roc_auc_score(rise >= t, da.risk)),
                         "spike_rate": float((rise >= t).mean())} for t in (30, 50, 70)}  # fmt: skip
    sc = p_oof[p_oof.scored]
    r5 = {"agreement": bin_agreement(sc.photo_g.to_numpy(),
                                     sc[["p_bin0", "p_bin1", "p_bin2", "p_bin3"]].to_numpy()),
          "old_bar": 0.90}  # fmt: skip
    res = {"what": "the intermediate's promised comparisons, run once against the frozen evidence "
                   "(pre-registered 8 Nov in docs/final_design.md); descriptive only",
           "r1_gbm_same_inputs": r1, "r2_class_weighting": r2, "r3_label_sensitivity": r3,
           "r4_branch_masking": r4, "r5_head_agreement": r5,
           "r6_missing_inputs": run_missing_inputs(ds, sp, cfg)}  # fmt: skip
    (OUT / "promised.json").write_text(json.dumps(res, indent=2))
    log.info("R1 LR-GBM {:+.3f} ({:+.3f},{:+.3f}) -> {}; R2 weighted-LR {:+.3f} ({:+.3f},{:+.3f}); R3 "
             "AUROC 30/50/70 {:.3f}/{:.3f}/{:.3f}; R4 masked-vs-no-meal {:+.4f}; R5 agreement {:.0%} -> {}",
             r1["point"], *r1["ci95"], r1["verdict"], r2["point"], *r2["ci95"],
             r3["spike_30"]["auroc_personal"], r3["spike_50"]["auroc_personal"],
             r3["spike_70"]["auroc_personal"], r4["point"], r5["agreement"], OUT / "promised.json")  # fmt: skip
    return res


def run_missing_inputs(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    """R6: missing heart rate and missing labs through the per-fold pipelines, reported in the
    originally promised metric (delta AUROC, delta ECE) next to the decision-flip rates."""
    from glimpse.calibration import choose_m_and_cutoffs
    from glimpse.demo_cases import meal_input
    from glimpse.evaluation.metrics import apply_platt, fit_platt
    from glimpse.evaluation.run import fold_estimates, fold_fusion, inner_oof
    from glimpse.serve import Pipeline
    from glimpse.studies.robustness import _flips
    from glimpse.training.final import cost_dict

    log, pcv, y = logger.bind(stage="promised"), Path(cfg.paths.outputs) / "photo_cv", ds.labels()
    oof, nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    nan = float("nan")
    base, no_hr, no_labs, ys = [], [], [], []
    for k, held in enumerate(sp.dev_folds):
        kk = k + 1
        pt = photo_table(nested[nested.fold == kk]) | photo_table(oof[oof.fold == kk])
        lr, c = fold_fusion(ds, sp, cfg, k, pt)
        rows_i, p_i = inner_oof(ds, sp, cfg, k, pt, c)
        a_k, b_k = fit_platt(p_i, y[rows_i])
        if a_k <= 0:
            a_k, b_k = 1.0, 0.0
        times_i = np.array([ds.samples[i].meal_time for i in rows_i], dtype="datetime64[s]")
        person_i = np.array([ds.samples[i].person_id for i in rows_i])
        sel = choose_m_and_cutoffs(apply_platt(p_i, a_k, b_k), y[rows_i], person_i, times_i, cost_dict(cfg))
        tab = Pipeline(cfg, photo_table=fold_estimates(oof[oof.fold == kk]), fusion=lr,
                       calibration=(a_k, b_k), cutoffs=(sel["low_risk_below"], sel["caution_from"]))  # fmt: skip
        for row in meal_rows(ds, set(held), scored_only=True):
            m = meal_input(cfg, ds, int(row))
            base.append(tab.answer(m, key=int(row)))
            no_hr.append(tab.answer(m.model_copy(update={"hr_times": [], "hr_bpm": []}), key=int(row)))
            no_labs.append(tab.answer(m.model_copy(
                update={"profile": m.profile | {"a1c_pct": nan, "fasting_glucose_mgdl": nan}}), key=int(row)))
            ys.append(int(y[int(row)]))
        log.info("R6 fold {} answered", kk)
    yv = np.array(ys)
    out = {}
    for name, pert in (("missing_hr", no_hr), ("missing_labs", no_labs)):
        both = [i for i in range(len(base)) if base[i].risk is not None and pert[i].risk is not None]
        pb = np.array([base[i].risk for i in both])
        pp = np.array([pert[i].risk for i in both])
        d_auroc = float(roc_auc_score(yv[both], pp) - roc_auc_score(yv[both], pb)) if both else None
        d_ece = float(ece_equal_mass(pp, yv[both]) - ece_equal_mass(pb, yv[both])) if both else None
        out[name] = {"answered_share": float(np.mean([a.risk is not None for a in pert])),
                     "co_answered": len(both), "delta_auroc": d_auroc, "delta_ece": d_ece,
                     "bar": "delta AUROC >= -0.03 (intermediate line 138)",
                     "flips": _flips([a.decision for a in base], [a.decision for a in pert])}  # fmt: skip
        log.info("R6 {}: answered {:.0%}, delta AUROC {} (bar -0.03), delta ECE {}", name,
                 out[name]["answered_share"], d_auroc, d_ece)  # fmt: skip
    return out
