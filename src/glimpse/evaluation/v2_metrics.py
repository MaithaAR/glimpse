"""v2 (sensor first) beside v1 on development (`glimpse v2-metrics`, 26 Nov; docs/v2_sensor_first.md).

Reads the committed v2 and v1 development answers and writes outputs/report_metrics/v2_metrics.json: the
registered endpoints (discrimination, calibration, usefulness with intervals, paired v2 minus v1, answered
share) and the same descriptive cuts the report gives v1 (served point without history, the samples' path
without priors, within-person AUROC, glycaemic status, the abstention band). The samples' path is exact here:
each fold's v2 fusion is refitted by the committed recipe and scored with the onboarding priors removed.
Nothing is chosen; the locked eight are not touched."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss, roc_auc_score

from glimpse.evaluation.metrics import (
    expected_cost,
    paired_person_bootstrap,
    person_bootstrap_ci,
    usefulness_report,
)
from glimpse.evaluation.report_metrics import ACTION, policies, scores
from glimpse.training.final import cost_dict

V1 = Path("outputs/evaluation_v2/dev_answers.csv")
V2 = Path("outputs/evaluation_sensor_first")
OUT = Path("outputs/report_metrics/v2_metrics.json")


def _ci(sids, fn) -> dict:
    pt, lo, hi = person_bootstrap_ci(sids, fn)
    return {"point": pt, "ci95": [lo, hi]}


def _cost_reduction(dec: pd.Series, y: np.ndarray, costs: dict) -> float:
    best = min(expected_cost(np.zeros_like(y), y, costs), expected_cost(np.ones_like(y), y, costs))
    return 1 - expected_cost(dec.map(ACTION).to_numpy(), y, costs) / best


def decisions(df: pd.DataFrame, costs: dict, targets: dict) -> dict:
    """The four usefulness criteria with person-bootstrap intervals (the protocol's ask)."""
    y, sids, dec = df.spike.to_numpy().astype(int), df.person_id.to_numpy(), df.decision
    rep = usefulness_report(dec.map(ACTION).to_numpy(), y, costs, targets)
    cl = (dec == "low risk").to_numpy()
    rep["cost_reduction_ci"] = _ci(sids, lambda i: _cost_reduction(dec.iloc[i].reset_index(drop=True), y[i], costs))
    rep["cleared_ci"] = _ci(sids, lambda i: float(cl[i].mean()))
    rep["cant_judge_ci"] = _ci(sids, lambda i: float((dec.to_numpy()[i] == "can't judge").mean()))
    rep["missed_ci"] = _ci(sids, lambda i: float(y[i][cl[i]].mean()) if cl[i].any() else float("nan"))
    rep["cleared_n"], rep["missed_n"] = int(cl.sum()), int(y[cl].sum())
    rep["people_with_a_clear"] = int(df.person_id[cl].nunique())
    rep["policies"] = policies(df, costs)
    return rep


def paired(v1: pd.DataFrame, v2: pd.DataFrame) -> dict:
    """v2 minus v1 on the meals both answer with a risk (registered endpoint c; margin 0.03)."""
    m = v1[["row", "risk"]].merge(v2[["row", "risk", "spike", "person_id"]], on="row", suffixes=("_v1", "_v2"))
    m = m.dropna(subset=["risk_v1", "risk_v2"])
    y, s = m.spike.to_numpy().astype(int), m.person_id.to_numpy()
    d, lo, hi = paired_person_bootstrap(y, m.risk_v2.to_numpy(), m.risk_v1.to_numpy(), s)
    ll = person_bootstrap_ci(
        s,
        lambda i: float(
            log_loss(y[i], m.risk_v1.to_numpy()[i], labels=[0, 1])
            - log_loss(y[i], m.risk_v2.to_numpy()[i], labels=[0, 1])
        ),
    )
    return {"meals": int(len(m)), "auroc_v1": float(roc_auc_score(y, m.risk_v1)),
            "auroc_v2": float(roc_auc_score(y, m.risk_v2)),
            "v2_minus_v1": float(roc_auc_score(y, m.risk_v2) - roc_auc_score(y, m.risk_v1)),
            "bootstrap_mean": d, "ci95": [lo, hi],  # 26 Nov fix: observed point, as v1's paired rows report
            "noninferior_at_0.03": bool(lo > -0.03),
            "log_loss_v1_minus_v2": {"point": ll[0], "ci95": [ll[1], ll[2]]}}  # fmt: skip


def within_person(df: pd.DataFrame) -> dict:
    a = df.dropna(subset=["risk"])
    raw, sh = [], []
    for _, g in a.groupby("person_id"):
        if g.spike.nunique() == 2:
            raw.append(roc_auc_score(g.spike, g.risk_raw))
            sh.append(roc_auc_score(g.spike, g.risk))
    d = np.array(sh) - np.array(raw)
    rng = np.random.default_rng(0)
    boot = [float(rng.choice(d, len(d)).mean()) for _ in range(2000)]
    return {"people": len(d), "raw": float(np.mean(raw)), "shipped": float(np.mean(sh)),
            "shipped_minus_raw": float(d.mean()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
            "people_worse": int((d < 0).sum())}  # fmt: skip


def subgroups(df: pd.DataFrame) -> dict:
    group = pd.read_csv("outputs/cv/oof_predictions.csv").groupby("person_id").group.first()
    out = {}
    for g, q in df.assign(group=df.person_id.map(group)).groupby("group"):
        a = q.dropna(subset=["risk"])
        cl = q[q.decision == "low risk"]
        out[str(g)] = {"people": int(q.person_id.nunique()), "meals": int(len(q)),
                       "spike_rate": float(q.spike.mean()), "auroc": float(roc_auc_score(a.spike, a.risk)),
                       "mean_risk_minus_spike_rate": float(a.risk.mean() - a.spike.mean()),
                       "cleared_share": float(len(cl) / len(q)), "cleared": int(len(cl)),
                       "missed": int(cl.spike.sum())}  # fmt: skip
    return out


def band(df: pd.DataFrame, costs: dict) -> dict:
    """What the abstention band does: its meals, their spike rate, and the cost if they were decided."""
    folds = json.loads((V2 / "dev_summary.json").read_text())["folds"]
    cut = {f["fold"]: (f["low_risk_below"], f["caution_from"]) for f in folds}
    a = df.dropna(subset=["risk"]).copy()
    lo, hi = a.fold.map(lambda k: cut[k][0]), a.fold.map(lambda k: cut[k][1])
    inband = (a.risk >= lo) & (a.risk < hi)
    y = a.spike.to_numpy().astype(int)
    decided = a.decision.where(~(inband & (a.decision == "can't judge")), "caution")
    return {"widths": {str(k): round(v[1] - v[0], 4) for k, v in cut.items()},
            "borderline_meals": int((inband & (a.decision == "can't judge")).sum()),
            "borderline_spike_rate": float(a.spike[inband & (a.decision == "can't judge")].mean())
            if (inband & (a.decision == "can't judge")).any() else None,
            "cost_shipped": expected_cost(a.decision.map(ACTION).to_numpy(), y, costs) / 1.0,
            "cost_band_decided_as_caution": expected_cost(decided.map(ACTION).to_numpy(), y, costs)}  # fmt: skip


def samples_path(cfg, df: pd.DataFrame, costs: dict) -> dict:
    """The grader's samples carry neither history nor onboarding priors. Exact on development: each fold's
    v2 fusion (refitted by the committed recipe, as the evaluation fitted it) scores its held-out meals with
    the two priors removed (the fold imputer's medians), then the fold's Platt map and the FROZEN v2
    cut-offs; reported before and after rule R13."""
    from glimpse.data.dataset import PROFILE, MealDataset
    from glimpse.data.splits import SplitsV2
    from glimpse.evaluation.run import fold_fusion
    from glimpse.serve.rules import decision
    from glimpse.training.baselines import trend_features
    from glimpse.training.final import load_thresholds

    ds = MealDataset.load(Path("data/dataset"))
    sp = SplitsV2.load(Path("splits/dev_splits_v2.json"))
    th = load_thresholds(Path(cfg.prototype.sensor_first_thresholds))["main"]
    lo, hi = th["low_risk_below"], th["caution_from"]
    folds = {f["fold"]: f for f in json.loads((V2 / "dev_summary.json").read_text())["folds"]}
    prior_cols = [7 + PROFILE.index("prior_spike_rate"), 7 + PROFILE.index("prior_rise")]
    risk = pd.Series(np.nan, index=df.row.astype(int))
    for k in range(len(sp.dev_folds)):
        lr, c = fold_fusion(ds, sp, cfg, k, {}, sensor_first=True)
        assert abs(c - folds[k + 1]["C"]) < 1e-12, "refit does not match the evaluation's fold fusion"
        rows = df[(df.fold == k + 1) & df.risk.notna()].row.astype(int).to_numpy()
        x = trend_features(ds, rows)
        x[:, prior_cols] = np.nan
        z = np.log(np.clip(lr.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6))
        z = z - np.log(np.clip(1 - lr.predict_proba(x)[:, 1], 1e-6, 1))
        f = folds[k + 1]["platt_slope"] * z + folds[k + 1]["platt_intercept"]
        risk.loc[rows] = 1 / (1 + np.exp(-f))
    r = risk.loc[df.row.astype(int)].to_numpy()
    has = ~np.isnan(r)
    dec = df.decision.copy()
    dec[has] = [decision(float(v), lo, hi) for v in r[has]]
    y = df.spike.to_numpy().astype(int)
    before = usefulness_report(dec.map(ACTION).to_numpy(), y, costs, cfg.targets.model_dump())
    before["n_cleared"], before["n_missed"] = int((dec == "low risk").sum()), int(df.spike[dec == "low risk"].sum())
    before["auroc"] = float(roc_auc_score(y[has], r[has]))
    after = dec.where(dec != "low risk", "can't judge")
    r13 = usefulness_report(after.map(ACTION).to_numpy(), y, costs, cfg.targets.model_dump())
    return {"cutoffs": [lo, hi], "before_r13": before, "with_r13": r13}


def run_v2_metrics(cfg) -> dict:
    costs, targets = cost_dict(cfg), cfg.targets.model_dump()
    v1, v2 = pd.read_csv(V1), pd.read_csv(V2 / "dev_answers.csv")
    summ = json.loads((V2 / "dev_summary.json").read_text())
    a = v2.dropna(subset=["risk"])
    y, s = a.spike.to_numpy().astype(int), a.person_id.to_numpy()
    res = {"what": __doc__.splitlines()[0], "meals": int(len(v2)), "answered_with_risk": int(len(a)),
           "v1_answered_with_risk": int(v1.risk.notna().sum()),
           "auroc": summ["risk_personal"]["auroc"], "auroc_raw": summ["risk_raw"]["auroc"]["point"],
           "auroc_platt_only": summ["risk_platt_only"]["auroc"]["point"],
           "auroc_excluding_fold_1": summ["risk_personal_excluding_fold1"]["auroc"],
           "per_fold": summ["per_fold_auroc"], "folds": [{k: f[k] for k in ("fold", "C", "platt_slope",
           "platt_intercept", "m", "low_risk_below", "caution_from")} for f in summ["folds"]],
           "shipped": scores(y, a.risk.to_numpy(float), s), "raw": scores(y, a.risk_raw.to_numpy(float), s),
           "platt_only": scores(y, a.risk_platt.to_numpy(float), s),
           "decisions": decisions(v2, costs, targets), "paired_v2_minus_v1": paired(v1, v2),
           "within_person": within_person(v2), "subgroups": subgroups(v2), "band": band(v2, costs),
           "cost_sweep": summ["cost_sweep"]}  # fmt: skip
    th = json.loads(Path(cfg.prototype.sensor_first_thresholds).read_text())
    lo, hi = th["main"]["low_risk_below"], th["main"]["caution_from"]
    from glimpse.serve.rules import decision

    dec = v2.decision.copy()
    has = v2.risk_platt.notna()
    dec[has] = [decision(float(v), lo, hi) for v in v2.risk_platt[has]]
    served = usefulness_report(dec.map(ACTION).to_numpy(), v2.spike.to_numpy().astype(int), costs, targets)
    served["n_cleared"], served["n_missed"] = int((dec == "low risk").sum()), int(v2.spike[dec == "low risk"].sum())
    res["served_no_history"] = served | {"cutoffs": [lo, hi]}
    res["samples_path"] = samples_path(cfg, v2, costs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    p = res["paired_v2_minus_v1"]
    logger.bind(stage="v2-metrics").info("v2 AUROC {:.3f} on {} meals; v2 - v1 {:+.3f} ({:+.3f}, {:+.3f}) on {}",
                                         res["auroc"]["point"], len(a), p["v2_minus_v1"], *p["ci95"],
                                         p["meals"])  # fmt: skip
    return res
