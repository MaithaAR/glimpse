"""Descriptive metrics re-derived from the committed answer files (`glimpse report-metrics`, 26 Nov).

Needs no dataset and no weights: it reads outputs/evaluation_v2/dev_answers.csv, the single locked run's
outputs/locked_eval/locked_answers.csv, outputs/cv/oof_predictions.csv and the photo-CV summaries, and
writes outputs/report_metrics/report_metrics.json. Nothing is refitted, no threshold is chosen and the
locked people are not rescored: every number is arithmetic on predictions already committed.

What it adds (each a finding of the 25 Nov cold grades): the protocol's registered AUPRC and Brier, ECE
under the protocol's equal-width bins next to the reported equal-mass bins with per-bin counts, the
baselines on the same answered meals, the RQ2 pair on partition v2, the cost of each constant policy,
the locked 150-of-153 reconciliation and the photo recipe's inner-fold evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from glimpse.evaluation.metrics import (
    ece_equal_mass,
    expected_cost,
    paired_person_bootstrap,
    person_bootstrap_ci,
    reliability,
)

OUT = Path("outputs/report_metrics")
DEV = Path("outputs/evaluation_v2/dev_answers.csv")
LOCKED = Path("outputs/locked_eval/locked_answers.csv")
ACTION = {"low risk": 0, "caution": 1, "can't judge": 2}


def _equal_mass_bins(p: np.ndarray, y: np.ndarray, bins: int = 15) -> list[dict]:
    """The bins behind ece_equal_mass, with their counts (same np.array_split rule)."""
    order = np.argsort(p)
    return [{"n": int(len(c)), "mean_predicted": float(p[c].mean()), "observed": float(y[c].mean())}
            for c in np.array_split(order, bins) if len(c)]  # fmt: skip


def scores(y: np.ndarray, p: np.ndarray, sids: np.ndarray) -> dict:
    """Discrimination and calibration for one risk column; intervals resample people."""
    auprc = person_bootstrap_ci(sids, lambda i: _safe(average_precision_score, y[i], p[i]))
    brier = person_bootstrap_ci(sids, lambda i: float(brier_score_loss(y[i], p[i])))
    rows, ece_w = reliability(p, y, bins=15)  # the registered binning: equal-width, targets.ece_bins = 15
    return {"n": int(len(y)), "base_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)),
            "auprc": {"point": auprc[0], "ci95": [auprc[1], auprc[2]]},
            "brier": {"point": brier[0], "ci95": [brier[1], brier[2]]},
            "brier_constant_base_rate": float(brier_score_loss(y, np.full_like(p, y.mean()))),
            "log_loss": float(log_loss(y, p)),
            "ece_equal_mass_15": ece_equal_mass(p, y, 15), "ece_equal_mass_10": ece_equal_mass(p, y, 10),
            "ece_equal_width_15": ece_w,
            "bins_equal_mass_15": _equal_mass_bins(p, y, 15),
            "bins_equal_width_15": [{"mean_predicted": m, "observed": o, "n": n} for m, o, n in rows]}  # fmt: skip


def _safe(fn, y, p) -> float:
    """fn(y, p), or NaN when a resample holds one class only (skipped by person_bootstrap_ci)."""
    return float(fn(y, p)) if 0 < y.sum() < len(y) else float("nan")


def policies(df: pd.DataFrame, costs: dict) -> dict:
    """Mean cost per meal of the shipped decisions and of each constant policy, on all meals of a cohort."""
    y = df.spike.to_numpy().astype(int)
    act = df.decision.map(ACTION).to_numpy()
    out = {"shipped": expected_cost(act, y, costs)}
    for name, a in (("always low risk", 0), ("always caution", 1), ("always can't judge", 2)):
        out[name] = expected_cost(np.full_like(y, a), y, costs)
    best = min(out["always low risk"], out["always caution"])
    out["best_constant"] = "always caution" if out["always caution"] <= out["always low risk"] else "always low risk"
    out["cost_reduction_vs_best_constant"] = 1 - out["shipped"] / best
    out["shares"] = {k: float((df.decision == k).mean()) for k in ACTION}
    return out


def baselines(dev: pd.DataFrame) -> dict:
    """The comparison models' out-of-fold risks (partition v2) on all 481 meals and on the 460 answered."""
    oof = pd.read_csv("outputs/cv/oof_predictions.csv")
    cnn = oof.groupby("row").sensor_cnn.mean()  # 3 seeds; the other models are deterministic
    base = oof[oof.seed == 0].set_index("row").assign(sensor_cnn_3seed_mean=cnn)
    answered = set(dev.loc[dev.risk.notna(), "row"].astype(int))
    out = {}
    for subset, rows in (("all_481", base.index), ("answered_460", [r for r in base.index if r in answered])):
        b = base.loc[rows]
        y = b.spike.to_numpy()
        out[subset] = {m: {"auroc": float(roc_auc_score(y, b[m])), "brier": float(brier_score_loss(y, b[m])),
                           "auprc": float(average_precision_score(y, b[m]))}
                       for m in MODELS}  # fmt: skip
    b = base
    d, lo, hi = paired_person_bootstrap(b.spike.to_numpy(), b.sensor_cnn_3seed_mean.to_numpy(),
                                        b.trend_lr.to_numpy(), b.person_id.to_numpy())  # fmt: skip
    point = float(roc_auc_score(b.spike, b.sensor_cnn_3seed_mean) - roc_auc_score(b.spike, b.trend_lr))
    out["rq2_partition_v2"] = {"sensor_cnn_minus_trend_lr": point, "bootstrap_mean": d, "ci95": [lo, hi],
                               "bar": "CNN beats the hand-made trends only if the lower bound > 0",
                               "verdict": "beats" if lo > 0 else "tie or loss"}  # fmt: skip
    return out


LAST = "inner_mae_last_stage_per_epoch_g"
MODELS = ("prevalence", "profile_lr", "trend_lr", "trend_gbm", "sensor_cnn_3seed_mean")


def photo_recipe() -> dict:
    """Per outer fold: heads-only vs heads + last stage inner-fold MAE behind the 12 + 2 recipe."""
    s = json.loads(Path("outputs/photo_cv/photo_cv_summary.json").read_text())
    rows = []
    for f in s["folds"]:
        k = f["last_stage_epochs"]
        rows.append({"fold": f["fold"], "heads_epochs": f["heads_epochs"],
                     "inner_mae_heads_only_g": float(np.mean(f["inner_mae_heads_only_g"])),
                     "inner_mae_with_last_stage_g": float(np.mean([e[k - 1] for e in f[LAST]])),
                     "last_stage_epochs": k, "last_stage_kept": f["last_stage_kept"],
                     "ridge_alpha": f["ridge_alpha"]})  # fmt: skip
    return {"folds": rows, "held_out": s["scored_meals_with_photo"]["methods"]}


def false_clears(dev: pd.DataFrame) -> dict:
    """Exploratory, chosen after the outcome: where the dev missed-among-cleared meals sit among all cleared
    meals on the fusion's own inputs (committed standardised features; percentile = share of cleared meals
    below). Near-cut misses are within 0.031 of their fold's low-risk cut, the report's T4/T5 boundary."""
    ev = json.loads(Path("outputs/evaluation_v2/dev_summary.json").read_text())
    cut = {f["fold"]: f["low_risk_below"] for f in ev["folds"]}
    z = pd.read_csv("outputs/figures/shap_features_z.csv").set_index("row")
    cleared = dev[dev.decision == "low risk"]
    miss = cleared[cleared.spike == 1].assign(gap=lambda d: d.fold.map(cut) - d.risk)
    cols = ["prior spike rate", "prior rise", "log photo grams", "window rise", "last glucose"]
    rows = []
    for _, m in miss.iterrows():
        pct = {c: float((z.loc[cleared.row, c] < z.loc[m.row, c]).mean()) for c in cols}
        rows.append({"row": int(m.row), "person_id": int(m.person_id), "fold": int(m.fold), "risk": float(m.risk),
                     "below_cut_by": float(m.gap), "n_history": int(m.n_history),
                     "class": "T4 near-cut" if m.gap <= 0.031 else "T5 confident",
                     "percentile_among_cleared": pct})  # fmt: skip
    return {"note": false_clears.__doc__.split("\n")[0], "meals": rows}


def paired_gains(views: dict) -> dict:
    """Shipped (personal) minus raw, paired by meal, intervals resampling people: the same standard of
    evidence the photo gate is held to (added 26 Nov after the second grading round)."""
    out = {}
    for name in ("dev", "locked"):
        a = views[name][views[name].risk.notna()]
        y, sids = a.spike.to_numpy().astype(int), a.person_id.to_numpy()
        p, r = a.risk.to_numpy(float), a.risk_raw.to_numpy(float)
        d, lo, hi = paired_person_bootstrap(y, p, r, sids)
        point = float(roc_auc_score(y, p) - roc_auc_score(y, r))
        brier = person_bootstrap_ci(sids, _diff(brier_score_loss, y, p, r))
        ll = person_bootstrap_ci(sids, _diff(log_loss, y, p, r))
        out[name] = {"auroc": {"point": point, "bootstrap_mean": d, "ci95": [lo, hi]},
                     "brier": {"point": brier[0], "ci95": [brier[1], brier[2]]},
                     "log_loss": {"point": ll[0], "ci95": [ll[1], ll[2]]}}  # fmt: skip
    folds = views["dev"].dropna(subset=["risk"]).groupby("fold")
    out["dev"]["folds_improved"] = int(sum(roc_auc_score(g.spike, g.risk) > roc_auc_score(g.spike, g.risk_raw)
                                           for _, g in folds))  # fmt: skip
    return out


def _diff(fn, y, p, r):
    """fn(shipped) - fn(raw) on resampled meal indices (a closure per cohort, so nothing leaks across loops)."""
    kw = {"labels": [0, 1]} if fn is log_loss else {}
    return lambda i: float(fn(y[i], p[i], **kw) - fn(y[i], r[i], **kw))


def brier_decomposition(y: np.ndarray, p: np.ndarray, bins: int = 15) -> dict:
    """Murphy: Brier ~ reliability - resolution + uncertainty, over the registered equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    base = y.mean()
    rel = sum((b == k).sum() * (p[b == k].mean() - y[b == k].mean()) ** 2 for k in range(bins) if (b == k).any())
    res = sum((b == k).sum() * (y[b == k].mean() - base) ** 2 for k in range(bins) if (b == k).any())
    return {"reliability": float(rel / len(y)), "resolution": float(res / len(y)),
            "uncertainty": float(base * (1 - base))}  # fmt: skip


def errors(df: pd.DataFrame) -> dict:
    """Every error class with counts: false clears, false cautions, spikes among can't-judge; who was cleared."""
    cleared = df[df.decision == "low risk"]
    caution = df[df.decision == "caution"]
    cj = df[df.decision == "can't judge"]
    return {"false_clears": int(cleared.spike.sum()), "cleared": len(cleared),
            "false_cautions": int((caution.spike == 0).sum()), "cautions": len(caution),
            "cant_judge_spiked": int(cj.spike.sum()), "cant_judge": len(cj),
            "people": int(df.person_id.nunique()), "people_with_a_clear": int(cleared.person_id.nunique()),
            "people_with_a_missed_spike": int(cleared[cleared.spike == 1].person_id.nunique()),
            "clears_by_person": {str(k): int(v) for k, v in cleared.person_id.value_counts().items()}}  # fmt: skip


def d8_paired() -> dict:
    """D8 against its control, paired: the photo arm trained without exact-macro matches vs the photo arm
    trained after a size-matched random drop, on the same photo meals (the no-meal arm cancels)."""
    d = pd.read_csv("outputs/ablations/d8_profile_excluded_oof.csv")
    d = d[d.has_photo]
    y, sids = d.spike.to_numpy().astype(int), d.person_id.to_numpy()
    m, lo, hi = paired_person_bootstrap(y, d.photo.to_numpy(), d.photo_random_drop.to_numpy(), sids)
    point = float(roc_auc_score(y, d.photo) - roc_auc_score(y, d.photo_random_drop))
    return {"matched_removed_minus_random_drop": point, "bootstrap_mean": m, "ci95": [lo, hi], "meals": len(d)}


def shap_terms(rows: list[int]) -> dict:
    """Centred linear SHAP terms of the frozen fusion (figures.json coefficients x committed standardised
    features) for chosen meals; the base value is the mean logit. Same arithmetic as Figure 6."""
    fig = json.loads(Path("outputs/figures/figures.json").read_text())["shap"]
    z = pd.read_csv("outputs/figures/shap_features_z.csv").set_index("row")
    names = list(fig["coefs"])
    coef = np.array([fig["coefs"][n] for n in names])
    centred = (z[names] - z[names].mean()) * coef
    corr = z[names].corr()
    return {"base_logit": fig["base_logit"],
            "meals": {str(r): {n: float(v) for n, v in centred.loc[r].items()} for r in rows},
            "heart_rate_correlations": {n: float(corr.loc["heart rate", n]) for n in names if n != "heart rate"},
            "photo_block_mean_abs": float(np.abs(centred[names[-4:]].sum(axis=1)).mean())}  # fmt: skip


def fine_tuned_parameters(cfg) -> dict:
    """Trainable parameters of the shipped photo model: the last ConvNeXt stage plus the two heads. The
    architecture is built without weights (no download); the head count is the budget file's photo total
    minus the backbone."""
    import timm

    m = timm.create_model(cfg.photo.arch, pretrained=False, num_classes=0)
    backbone = sum(p.numel() for p in m.parameters())
    last = sum(p.numel() for p in m.stages[-1].parameters())
    budget = pd.read_csv("outputs/param_budget.csv")
    photo = int(budget.loc[budget.model.str.startswith("PhotoCarbModel"), "parameters"].iloc[0])
    return {"backbone": backbone, "last_stage": last, "heads": photo - backbone,
            "fine_tuned_total": last + photo - backbone}  # fmt: skip


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def _served(df: pd.DataFrame, platt: np.ndarray, lo: float, hi: float, costs: dict, targets: dict) -> dict:
    """Decisions at the deployed cut-offs on a no-history (Platt-only) risk; rows without a risk stay can't
    judge. This is the operating point the folder command and the app serve."""
    from glimpse.evaluation.metrics import usefulness_report
    from glimpse.serve.rules import decision

    dec = df.decision.copy()
    has = ~np.isnan(platt)
    dec[has] = [decision(float(r), lo, hi) for r in platt[has]]
    rep = usefulness_report(dec.map(ACTION).to_numpy(), df.spike.to_numpy().astype(int), costs, targets)
    rep["n_missed"] = int(df.spike[dec == "low risk"].sum())
    return rep


def served_operating_point(cfg, dev: pd.DataFrame, locked: pd.DataFrame, costs: dict) -> dict:
    """The no-history operating point at the deployed cut-offs, descriptive (26 Nov): dev uses each fold's
    Platt-only risk (per-fold models), locked applies the frozen Platt map to the frozen fusion's raw risk.
    Nothing is refitted; the locked people are not rescored (arithmetic on the committed answers)."""
    import joblib

    from glimpse.training.final import FINAL, load_thresholds

    th = load_thresholds()["main"]
    lo, hi = th["low_risk_below"], th["caution_from"]
    a, b = joblib.load(FINAL)["calibration"]
    lk_platt = 1 / (1 + np.exp(-(a * _logit(locked.risk_raw.to_numpy(float)) + b)))
    lk_platt[locked.risk_raw.isna().to_numpy()] = np.nan
    targets = cfg.targets.model_dump()
    return {"cutoffs": [lo, hi], "platt": [a, b],
            "dev": _served(dev, dev.risk_platt.to_numpy(float), lo, hi, costs, targets),
            "locked": _served(locked, lk_platt, lo, hi, costs, targets)}  # fmt: skip


def ece_noise_floor(y: np.ndarray, p: np.ndarray, sids: np.ndarray, draws: int = 2000, seed: int = 0) -> dict:
    """What ECE (15 equal-width bins) a perfectly calibrated model shows at this sample size: labels drawn as
    Bernoulli(risk). Plus a person-bootstrap interval on the observed ECE."""
    rng = np.random.default_rng(seed)
    sims = [reliability(p, (rng.random(len(p)) < p).astype(float), bins=15)[1] for _ in range(draws)]
    obs = person_bootstrap_ci(sids, lambda i: reliability(p[i], y[i], bins=15)[1])
    return {"observed": obs[0], "observed_ci95": [obs[1], obs[2]],
            "perfectly_calibrated_median": float(np.median(sims)),
            "perfectly_calibrated_q95": float(np.quantile(sims, 0.95))}  # fmt: skip


def coverage_risk(df: pd.DataFrame, cut: np.ndarray, costs: dict) -> list[dict]:
    """Selective decisions, descriptive: abstain on the share of answered meals whose risk lies closest to the
    low-risk cut, decide the rest at the cut, and report what the cleared meals and the cost do."""
    a = df[df.risk.notna()].reset_index(drop=True)
    y = a.spike.to_numpy().astype(int)
    r = a.risk.to_numpy(float)
    dist = np.abs(r - cut[a.index] if cut.ndim else r - cut)
    order = np.argsort(dist)
    rows = []
    for share in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        act = np.where(r < (cut[a.index] if cut.ndim else cut), 0, 1)
        act[order[: int(round(share * len(a)))]] = 2
        cl = act == 0
        rows.append({"abstain_share": share, "cleared_share": float(cl.mean()),
                     "missed_among_cleared": float(y[cl].mean()) if cl.any() else None,
                     "error_rate_answered": float(np.mean((act[act < 2] == 0) == (y[act < 2] == 1))),
                     "cost": expected_cost(act, y, costs)})  # fmt: skip
    return rows


def person_offsets(dev: pd.DataFrame, locked: pd.DataFrame, platt: tuple[float, float]) -> dict:
    """The shipped risk's per-person part, b = logit(risk) - logit(Platt-only risk), from the answer files."""
    out = {}
    lk_platt = 1 / (1 + np.exp(-(platt[0] * _logit(locked.risk_raw.to_numpy(float)) + platt[1])))
    for name, risk, base in (("dev", dev.risk.to_numpy(float), dev.risk_platt.to_numpy(float)),
                             ("locked", locked.risk.to_numpy(float), lk_platt)):  # fmt: skip
        b = _logit(risk) - _logit(base)
        b = b[~np.isnan(b)]
        out[name] = {"median": float(np.median(b)), "q10": float(np.quantile(b, 0.1)),
                     "q90": float(np.quantile(b, 0.9)),
                     "share_abs_above_0_5": float(np.mean(np.abs(b) > 0.5))}  # fmt: skip
    return out


def heart_rate_r2() -> float:
    """R^2 of heart rate (standardised fusion input) regressed on the other 18 inputs, development meals."""
    from sklearn.linear_model import LinearRegression

    z = pd.read_csv("outputs/figures/shap_features_z.csv").drop(columns="row")
    x, y = z.drop(columns="heart rate"), z["heart rate"]
    return float(LinearRegression().fit(x, y).score(x, y))


def run_report_metrics(cfg) -> dict:
    """Write outputs/report_metrics/report_metrics.json; log the headline additions."""
    from glimpse.training.final import cost_dict

    costs = cost_dict(cfg)
    dev, locked = pd.read_csv(DEV), pd.read_csv(LOCKED)
    res: dict = {"what": __doc__.splitlines()[0], "cohorts": {}}
    views = {"dev": dev, "dev_excluding_fold1": dev[dev.fold != 1], "locked": locked}
    for name, df in views.items():
        a = df[df.risk.notna()]
        y, sids = a.spike.to_numpy().astype(int), a.person_id.to_numpy()
        cols = ["risk_raw", "risk_platt", "risk"] if "risk_platt" in a else ["risk_raw", "risk"]
        res["cohorts"][name] = {c: scores(y, a[c].to_numpy(float), sids) for c in cols}
        res["cohorts"][name]["decisions"] = policies(df, costs)
    res["ece_target"] = {
        "registered": f"ECE <= {cfg.targets.ece_max}, equal-width (protocol), {cfg.targets.ece_bins} bins (config)",
        **{
            f"{k}_{b}": res["cohorts"][k]["risk"][f"ece_{b}_15"]
            for k in ("dev", "locked")
            for b in ("equal_width", "equal_mass")
        },
    }
    cj = locked[locked.decision == "can't judge"]
    res["locked_reconciliation"] = {"meals": len(locked), "with_risk": int(locked.risk.notna().sum()),
                                    "cant_judge": len(cj), "cant_judge_with_risk": int(cj.risk.notna().sum()),
                                    "cant_judge_reasons": cj.reason.value_counts().to_dict()}  # fmt: skip
    res["baselines"] = baselines(dev)
    res["photo_recipe"] = photo_recipe()
    res["false_clears_exploratory"] = false_clears(dev)
    res["paired_personal_minus_raw"] = paired_gains(views)
    res["brier_decomposition"] = {k: {c: brier_decomposition(v[v.risk.notna()].spike.to_numpy().astype(float),
                                                             v.loc[v.risk.notna(), c].to_numpy(float))
                                      for c in ("risk_raw", "risk")}
                                  for k, v in (("dev", dev), ("locked", locked))}  # fmt: skip
    res["errors"] = {"dev": errors(dev), "locked": errors(locked)}
    res["d8_paired"] = d8_paired()
    t5 = [m["row"] for m in res["false_clears_exploratory"]["meals"] if m["class"].startswith("T5")]
    res["shap_terms_t5"] = shap_terms(t5)
    res["fine_tuned_parameters"] = fine_tuned_parameters(cfg)
    res["served_operating_point"] = served_operating_point(cfg, dev, locked, costs)
    res["ece_noise_floor"] = {}
    for k, v in (("dev", dev), ("locked", locked)):
        a = v[v.risk.notna()]
        res["ece_noise_floor"][k] = ece_noise_floor(a.spike.to_numpy().astype(float), a.risk.to_numpy(float),
                                                    a.person_id.to_numpy())  # fmt: skip
    ev = json.loads(Path("outputs/evaluation_v2/dev_summary.json").read_text())
    lo_by_fold = {f["fold"]: f["low_risk_below"] for f in ev["folds"]}
    dev_ans = dev[dev.risk.notna()].reset_index(drop=True)
    res["coverage_risk"] = {"dev": coverage_risk(dev_ans, dev_ans.fold.map(lo_by_fold).to_numpy(float), costs),
                            "locked": coverage_risk(locked, np.array(res["served_operating_point"]["cutoffs"][0]),
                                                    costs)}  # fmt: skip
    res["person_offsets"] = person_offsets(dev, locked, tuple(res["served_operating_point"]["platt"]))
    res["heart_rate_r2_from_other_inputs"] = heart_rate_r2()
    res["calibration_in_the_large_pooled"] = {k: float(v.risk.mean() - v.loc[v.risk.notna(), "spike"].mean())
                                              for k, v in (("dev", dev), ("locked", locked))}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report_metrics.json").write_text(json.dumps(res, indent=2))
    c = res["cohorts"]
    logger.bind(stage="report-metrics").info(
        "AUPRC dev {:.3f} (base {:.3f}) locked {:.3f} (base {:.3f}); Brier dev {:.3f} locked {:.3f}; "
        "ECE equal-width dev {:.3f}; RQ2 v2 {:+.3f}", c["dev"]["risk"]["auprc"]["point"], c["dev"]["risk"]["base_rate"],
        c["locked"]["risk"]["auprc"]["point"], c["locked"]["risk"]["base_rate"], c["dev"]["risk"]["brier"]["point"],
        c["locked"]["risk"]["brier"]["point"], c["dev"]["risk"]["ece_equal_width_15"],
        res["baselines"]["rq2_partition_v2"]["sensor_cnn_minus_trend_lr"])  # fmt: skip
    return res
