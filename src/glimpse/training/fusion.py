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
from glimpse.data.splits import SplitsV2, assert_not_locked, meal_rows
from glimpse.evaluation import noninferiority, paired_person_bootstrap
from glimpse.training.baselines import logistic, trend_features
from glimpse.training.cv import _tune_sklearn


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


# ---------------------------------------------------------------- Step 4c-3: fusion with the photo estimate
PHOTO_COLS = ["log_photo_g", "p_bin1", "p_bin2", "p_bin3"]  # p_bin0 left out: the four probabilities sum to 1
FUSION_ARMS = ["no_meal", "photo", "logged_carbs", "photo_shuffled", "ridge_photo", "photo_grams_only"]
# added after seeing results; reported as secondary, never as the main result. ridge_photo: after 4c-2.
# photo_grams_only (after Step 5): the fine-tuned photo estimate WITHOUT the bin probabilities, to show what the
# carb-bin head adds (the instructor asked for that head).
POST_HOC = ["ridge_photo", "photo_grams_only"]


def photo_table(df: pd.DataFrame) -> dict[int, list[float]]:
    """Meal row -> photo inputs for the fusion: log(1 + estimated grams) and the bin probabilities."""
    return {int(r.row): [float(np.log1p(r.photo_g)), r.p_bin1, r.p_bin2, r.p_bin3] for r in df.itertuples()}


def ridge_table(df: pd.DataFrame) -> dict[int, list[float]]:
    """Meal row -> log(1 + ridge estimate in grams)."""
    return {int(r.row): [float(np.log1p(r.ridge_g))] for r in df.itertuples()}


def with_table(table: dict[int, list[float]], width: int) -> Callable[[MealDataset, np.ndarray], np.ndarray]:
    """Trend + profile inputs plus looked-up meal columns; NaN for meals without a photo (training median later)."""
    nan = [float("nan")] * width
    return lambda ds, rows: np.hstack([trend_features(ds, rows), np.array([table.get(int(i), nan) for i in rows])])


def shuffle_within(x: np.ndarray, cols: int, seed: int) -> np.ndarray:
    """Control: the last `cols` columns (the photo inputs) swapped between meals; everything else unchanged."""
    out = x.copy()
    out[:, -cols:] = x[np.random.default_rng(seed).permutation(len(x)), -cols:]
    return out


def _slope(est: np.ndarray, true: np.ndarray) -> float:
    """Calibration slope on the log scale: 1 = estimates spread like the truth; below 1 = too spread out."""
    return float(np.polyfit(np.log1p(est), np.log1p(true), 1)[0])


def run_fusion(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    """Profile + trend logistic regression with each kind of meal information on partition v2 -- the same
    folds and inner folds as the photo CV and the evaluation; gains with person-paired 95% ranges."""
    from glimpse.training.photo_cv import ridge_estimates

    log, out = logger.bind(stage="fusion"), Path(cfg.paths.outputs) / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    pcv = Path(cfg.paths.outputs) / "photo_cv"
    if not (pcv / "photo_oof.csv").is_file():
        raise FileNotFoundError("run `just photo-cv` first")
    p_oof, p_nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    r_oof, r_nested, ridge_info = ridge_estimates(ds, sp, cfg)
    y, grid = ds.labels(), cfg.baselines.logreg_c_grid
    rows_out, tuning = [], []
    for k, fold in enumerate(sp.dev_folds):
        kk = k + 1
        people = sp.training_people(k)
        assert_not_locked(list(people) + list(fold), sp.locked, f"fusion fold {kk}")
        tr, te = meal_rows(ds, set(people), scored_only=False), meal_rows(ds, set(fold), scored_only=True)
        assert not {ds.samples[i].person_id for i in tr} & set(fold), "a held-out person is in training"
        # training side = nested estimates (model never saw that person); test side = out-of-fold estimates
        pt = photo_table(p_nested[p_nested.fold == kk]) | photo_table(p_oof[p_oof.fold == kk])
        rt = ridge_table(r_nested[r_nested.fold == kk]) | ridge_table(r_oof[r_oof.fold == kk])
        inner = sp.inner_folds[k]  # the one validation split (D1): target holds only
        gt = {r: v[:1] for r, v in pt.items()}  # log grams only, no bin probabilities
        X = {"no_meal": trend_features, "photo": with_table(pt, len(PHOTO_COLS)),
             "logged_carbs": arm_features("logged_carbs"), "ridge_photo": with_table(rt, 1),
             "photo_grams_only": with_table(gt, 1)}  # fmt: skip
        preds, chosen = {}, {"fold": kk}
        for arm, fx in X.items():
            c, _ = _tune_sklearn(lambda c: logistic(c, cfg.seed), grid, fx, ds, people, inner, y)
            chosen[f"{arm}_C"] = c
            m = logistic(c, cfg.seed).fit(fx(ds, tr), y[tr])
            preds[arm] = m.predict_proba(fx(ds, te))[:, 1]
            if arm == "photo":  # control: same fitted model, photo inputs swapped between this fold's test meals
                xs = shuffle_within(fx(ds, te), len(PHOTO_COLS), cfg.seed + kk)
                preds["photo_shuffled"] = m.predict_proba(xs)[:, 1]
        tuning.append(chosen)
        for j, i in enumerate(te):
            s = ds.samples[i]
            rows_out.append({"fold": kk, "row": int(i), "person_id": s.person_id, "spike": int(y[i]),
                             "has_photo": int(i) in pt, **{a: float(preds[a][j]) for a in FUSION_ARMS}})  # fmt: skip
    oof = pd.DataFrame(rows_out)
    oof.to_csv(out / "fusion_oof.csv", index=False)
    res = summarise_fusion(oof, cfg) | {"tuning": tuning, "ridge_posthoc": ridge_info}
    scored_photo = p_oof[p_oof.scored]
    res["photo_estimate_calibration_slope"] = {"out_of_fold": _slope(scored_photo.photo_g, scored_photo.carbs_g),
                                               "nested": _slope(p_nested.photo_g, p_nested.carbs_g)}  # fmt: skip
    rs = r_oof[r_oof.row.isin(scored_photo.row)]
    res["ridge_posthoc_mae_g"] = float(np.mean(np.abs(rs.ridge_g - rs.carbs_g)))
    (out / "fusion_summary.json").write_text(json.dumps(res, indent=2))
    for a in FUSION_ARMS:
        log.info("{:<15} AUROC pooled {:.3f} | within folds {:.3f}{}", a, res["auroc_pooled"][a],
                 res["auroc_within_fold_mean"][a], "  (post-hoc)" if a in POST_HOC else "")  # fmt: skip
    for name, v in res["gains_vs_no_meal"].items():
        log.info("gain {:<15} {:+.3f} (95% range {:+.3f} to {:+.3f})", name, v["point"], *v["ci95"])
    pm = res["primary_photo_meals_only"]
    log.info("PRIMARY photo gain, meals with a photo only: {:+.3f} (95% range {:+.3f} to {:+.3f})", pm["point"],
             *pm["ci95"])  # fmt: skip
    rq = res["rq1_photo_vs_logged"]
    log.info("RQ1 photo - logged carbs {:+.3f} (95% range {:+.3f} to {:+.3f}) -> {}", rq["point"], *rq["ci95"],
             rq["verdict"])  # fmt: skip
    log.info("post-hoc ridge carb error {:.1f} g; photo estimate calibration slope OOF {:.2f}, nested {:.2f}",
             res["ridge_posthoc_mae_g"], *res["photo_estimate_calibration_slope"].values())  # fmt: skip
    return res


def summarise_fusion(oof: pd.DataFrame, cfg: GlimpseConfig) -> dict:
    """AUROC per arm (pooled = primary, within folds = check); every gain vs no meal with a paired person bootstrap;
    the primary gain again on meals with a photo; RQ1 (photo vs logged carbs) against the proposal's margin."""
    yy, sids, B, seed = oof.spike.to_numpy(), oof.person_id.to_numpy(), cfg.fusion.bootstrap_draws, cfg.seed

    def gain(a: str, b: str, df: pd.DataFrame = oof) -> dict:
        """AUROC of arm a minus arm b on these meals, with a paired 95% range by person."""
        y_, s_ = df.spike.to_numpy(), df.person_id.to_numpy()
        mean, lo, hi = paired_person_bootstrap(y_, df[a].to_numpy(), df[b].to_numpy(), s_, B=B, seed=seed)
        return {"point": _auc(y_, df[a].to_numpy()) - _auc(y_, df[b].to_numpy()), "bootstrap_mean": mean,
                "ci95": [lo, hi]}  # fmt: skip

    def logloss_secondary(df: pd.DataFrame) -> dict:
        """Secondary endpoint (D7, added at the final stage and labelled as such): paired per-meal log-loss,
        photo vs no-meal, clustered by person, one-sided at 5%. Positive = the photo arm is better."""
        eps, y_ = 1e-6, df.spike.to_numpy().astype(float)
        la = -(y_ * np.log(np.clip(df.photo, eps, 1 - eps)) + (1 - y_) * np.log(np.clip(1 - df.photo, eps, 1 - eps)))
        lb = -(y_ * np.log(np.clip(df.no_meal, eps, 1 - eps))
               + (1 - y_) * np.log(np.clip(1 - df.no_meal, eps, 1 - eps)))  # fmt: skip
        d, s_ = (lb - la).to_numpy(), df.person_id.to_numpy()
        ids, rng = np.unique(s_), np.random.default_rng(seed)
        where = {p: np.where(s_ == p)[0] for p in ids}
        stats = np.array([float(np.mean(d[np.concatenate([where[p] for p in rng.choice(ids, len(ids))])]))
                          for _ in range(B)])  # fmt: skip
        return {"point": float(d.mean()), "ci95": [float(np.percentile(stats, 2.5)),
                                                   float(np.percentile(stats, 97.5))],
                "p_one_sided": float(np.mean(stats <= 0))}  # fmt: skip

    rq1 = gain("photo", "logged_carbs")
    rq1["margin"] = cfg.targets.rq1_noninferiority_margin
    rq1["verdict"] = noninferiority(rq1["ci95"][0], rq1["ci95"][1], rq1["margin"])
    return {
        "scored_meals": int(len(oof)), "with_photo": int(oof.has_photo.sum()), "people": int(len(set(sids))),
        "spikes": int(yy.sum()),
        "auroc_pooled": {a: _auc(yy, oof[a].to_numpy()) for a in FUSION_ARMS},
        "auroc_within_fold_mean": {a: float(np.mean([_auc(f.spike.to_numpy(), f[a].to_numpy())
                                                     for _, f in oof.groupby("fold")])) for a in FUSION_ARMS},
        "gains_vs_no_meal": {a: gain(a, "no_meal") for a in FUSION_ARMS if a != "no_meal"},
        "primary_photo_meals_only": gain("photo", "no_meal", oof[oof.has_photo]),
        "secondary_logloss_photo_vs_no_meal": logloss_secondary(oof[oof.has_photo.astype(bool)]),
        "rq1_photo_vs_logged": rq1,
        "post_hoc_arms": POST_HOC,
    }  # fmt: skip
