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
