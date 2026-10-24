"""Fusion: one logistic regression on the profile + trend inputs, with or without meal information, on the same
5 outer person folds as the cross-validation (Step 3c).

Step 4c-1 is the logged-carbs check. The logged carbohydrates are the best meal information a photo could ever
give us, so the gain they bring is an upper bound for photo-estimated carbs (carbs only: a photo may also show fat
or fibre). If even the logged carbs cannot add a meaningful AUROC gain, a photo estimate cannot either."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits, meal_rows
from glimpse.evaluation import paired_person_bootstrap
from glimpse.training.baselines import logistic, trend_features
from glimpse.training.cv import _inner_folds, _tune_sklearn, outer_training_people

ARMS = ["no_meal", "logged_carbs"]


def logged_carbs(ds: MealDataset, rows: np.ndarray) -> np.ndarray:
    """log(1 + grams) of the logged carbohydrates, one column; NaN where none were logged (filled later with the
    training median). The log keeps a few very large meals from dominating."""
    return np.array([[np.log1p(c) if (c := ds.samples[i].carbs_served_g) is not None else np.nan] for i in rows])


def arm_features(arm: str) -> Callable[[MealDataset, np.ndarray], np.ndarray]:
    """Inputs of each arm: the trend model's inputs (7 trend numbers + 8 profile inputs), plus the meal column."""
    if arm == "no_meal":
        return trend_features
    if arm == "logged_carbs":
        return lambda ds, rows: np.hstack([trend_features(ds, rows), logged_carbs(ds, rows)])
    raise ValueError(f"unknown arm {arm}")


def carbs_check_decision(hi: float, min_gain: float) -> str:
    """Rule fixed before running: skip the photo CV only if the WHOLE 95% range of the gain is below min_gain
    (logged carbs clearly add too little). Any other result, including an unclear one, means proceed."""
    return "stop: carbs add too little" if hi < min_gain else "proceed to the photo cross-validation"


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else 0.5


def run_carbs_check(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> dict:
    """Out-of-fold predictions of both arms, AUROC per arm, the paired gain with its 95% range, and the decision."""
    log, out = logger.bind(stage="fusion"), Path(cfg.paths.outputs) / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    groups, y, grid = {s.person_id: s.group for s in ds.samples}, ds.labels(), cfg.baselines.logreg_c_grid
    rows_out, tuning = [], []
    for k, fold in enumerate(sp.outer_folds):
        people = outer_training_people(sp, k, groups)
        tr, te = meal_rows(ds, set(people), scored_only=False), meal_rows(ds, set(fold), scored_only=True)
        assert not {ds.samples[i].person_id for i in tr} & set(fold), "a held-out person is in training"
        inner = _inner_folds(people, groups, cfg.cv.inner_folds, cfg.seed + 100 + k)  # same inner folds as Step 3c
        preds, chosen = {}, {"fold": k + 1}
        for arm in ARMS:
            X = arm_features(arm)
            c, _ = _tune_sklearn(lambda c: logistic(c, cfg.seed), grid, X, ds, people, inner, y)
            chosen[f"{arm}_C"] = c
            if c in (min(grid), max(grid)):
                log.warning("fold {} {}: C={} is at the edge of the grid", k + 1, arm, c)
            preds[arm] = logistic(c, cfg.seed).fit(X(ds, tr), y[tr]).predict_proba(X(ds, te))[:, 1]
        tuning.append(chosen)
        for j, i in enumerate(te):
            s = ds.samples[i]
            rows_out.append({"fold": k + 1, "row": int(i), "person_id": s.person_id, "spike": int(y[i]),
                             "carbs_g": s.carbs_served_g, **{a: float(preds[a][j]) for a in ARMS}})  # fmt: skip
    oof = pd.DataFrame(rows_out)
    oof.to_csv(out / "carbs_check_oof.csv", index=False)

    yy, sids = oof.spike.to_numpy(), oof.person_id.to_numpy()
    carbs = oof.carbs_g.fillna(oof.carbs_g.median()).to_numpy()
    pooled = {a: _auc(yy, oof[a].to_numpy()) for a in ARMS}
    within = {a: float(np.mean([_auc(f.spike.to_numpy(), f[a].to_numpy()) for _, f in oof.groupby("fold")]))
              for a in ARMS}  # fmt: skip
    mean, lo, hi = paired_person_bootstrap(yy, oof.logged_carbs.to_numpy(), oof.no_meal.to_numpy(), sids,
                                           B=cfg.fusion.bootstrap_draws, seed=cfg.seed)  # fmt: skip
    res = {
        "scored_meals": len(oof), "people": int(oof.person_id.nunique()), "spikes": int(yy.sum()),
        "tuning": tuning, "auroc_pooled": pooled, "auroc_within_fold_mean": within,
        "auroc_carbs_alone_pooled": _auc(yy, carbs),
        "gain_logged_carbs": {"point": pooled["logged_carbs"] - pooled["no_meal"], "bootstrap_mean": mean,
                              "ci95": [lo, hi], "draws": cfg.fusion.bootstrap_draws},
        "min_gain": cfg.fusion.carbs_check_min_gain,
        "decision": carbs_check_decision(hi, cfg.fusion.carbs_check_min_gain),
        "note": "logged carbs = upper bound for photo-estimated carbs; the 95% range resamples people but not the "
                "inner-fold tuning, so it is slightly too narrow",
    }  # fmt: skip
    (out / "carbs_check.json").write_text(json.dumps(res, indent=2))
    for a in ARMS:
        log.info("{:<13} AUROC pooled {:.3f} | mean within folds {:.3f}", a, pooled[a], within[a])
    log.info("logged carbs alone: AUROC {:.3f}", res["auroc_carbs_alone_pooled"])
    log.info("gain from logged carbs {:+.3f} (95% range {:+.3f} to {:+.3f}, people resampled {} times)",
             res["gain_logged_carbs"]["point"], lo, hi, cfg.fusion.bootstrap_draws)  # fmt: skip
    log.info("decision (whole range below {} = stop): {}", cfg.fusion.carbs_check_min_gain, res["decision"])
    return res
