"""v2 explained, calibrated and its failures sorted (`glimpse v2-explain`; registered in docs/v2_explain.md).

Descriptive only. Reads the committed v2 development answers and the frozen v2 fusion; with data/ present
it also writes the fusion's standardised inputs to outputs/v2_explain/v2_features_z.csv once, so a clean
checkout redoes every number without the data. Nothing is chosen and nothing frozen changes."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from glimpse.evaluation.metrics import expected_cost, person_bootstrap_ci, reliability, usefulness_report
from glimpse.evaluation.report_metrics import ACTION, ece_noise_floor
from glimpse.training.final import cost_dict

V2 = Path("outputs/evaluation_sensor_first")
OUT = Path("outputs/v2_explain")
NAMES = ["last glucose", "slope 30 min", "slope 60 min", "window mean", "window sd", "window rise", "heart rate",
         "age", "BMI", "HbA1c", "fasting glucose", "hour (sin)", "hour (cos)", "prior spike rate",
         "prior rise"]  # fmt: skip
NEAR_CUT = 0.031  # the report's T4/T5 boundary, as for v1


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def features(cfg, rows: np.ndarray) -> pd.DataFrame:
    """The frozen v2 fusion's standardised inputs for the given development rows (committed csv if present)."""
    path = OUT / "v2_features_z.csv"
    if path.exists():
        z = pd.read_csv(path)
        assert (z.row.to_numpy() == rows).all(), "v2_features_z.csv rows differ from the v2 development answers"
        return z
    from glimpse.data.dataset import MealDataset
    from glimpse.training.baselines import trend_features

    ds = MealDataset.load(Path("data/dataset"))
    pipe = joblib.load(cfg.prototype.sensor_first_bundle)["models"]["sensor"]
    z = pd.DataFrame(pipe[:-1].transform(trend_features(ds, rows)), columns=NAMES)
    z.insert(0, "row", rows)
    z.insert(1, "meal_time", [str(ds.samples[int(r)].meal_time) for r in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    z.to_csv(path, index=False)
    return z


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def calibration(a: pd.DataFrame) -> dict:
    """E1: noise floor, person-bootstrap interval and the equal-width bins with counts and Wilson intervals."""
    y, p, s = a.spike.to_numpy().astype(int), a.risk.to_numpy(float), a.person_id.to_numpy()
    nf = ece_noise_floor(y, p, s)
    lo, hi = nf["observed_ci95_biased_upward"]
    verdict = "met" if hi <= 0.05 else ("on track" if nf["observed"] <= 0.05 else "not met")
    edges = np.linspace(0, 1, 16)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 14)
    bins = []
    for b in range(15):
        m = idx == b
        if m.any():
            k, n = int(y[m].sum()), int(m.sum())
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n, "mean_risk": float(p[m].mean()),
                         "spike_rate": k / n, "wilson95": wilson(k, n)})  # fmt: skip
    return nf | {"verdict": verdict, "bins": bins, "ece_check": float(reliability(p, y, bins=15)[1])}


def frozen_band(v2: pd.DataFrame, th: dict, costs: dict, targets: dict) -> dict:
    """E2: the frozen cut-offs on the development risks (in-sample: chosen on these people)."""
    lo, hi = th["main"]["low_risk_below"], th["main"]["caution_from"]
    dec = v2.decision.copy()
    has = v2.risk.notna()
    dec[has] = np.where(v2.risk[has] < lo, "low risk", np.where(v2.risk[has] >= hi, "caution", "can't judge"))
    y = v2.spike.to_numpy().astype(int)
    rep = usefulness_report(dec.map(ACTION).to_numpy(), y, costs, targets)
    band = has & (v2.risk >= lo) & (v2.risk < hi)
    cl = dec == "low risk"
    return rep | {"cutoffs": [lo, hi], "in_band": int(band.sum()), "in_band_share": float(band.mean()),
                  "in_band_spike_rate": float(v2.spike[band].mean()), "cleared": int(cl.sum()),
                  "missed": int(v2.spike[cl].sum())}  # fmt: skip


def miss_curve(a: pd.DataFrame) -> list[dict]:
    """E3: missed among cleared against share cleared as the low-risk cut moves (no band)."""
    out = []
    for c in np.round(np.arange(0.05, 0.301, 0.01), 2):
        cl = a.risk < c
        n = int(cl.sum())
        out.append({"cut": float(c), "cleared_share": float(cl.mean()), "cleared": n,
                    "missed_among_cleared": float(a.spike[cl].mean()) if n else None,
                    "missed_wilson95": wilson(int(a.spike[cl].sum()), n) if n else None})  # fmt: skip
    return out


def taxonomy(v2: pd.DataFrame, group: pd.Series, th: dict) -> dict:
    """E4: every v2 false clear sorted by the v1 rules."""
    folds = json.loads((V2 / "dev_summary.json").read_text())["folds"]
    cut = {f["fold"]: f["low_risk_below"] for f in folds}
    fc = v2[(v2.decision == "low risk") & (v2.spike == 1)].copy()
    fc["below_cut_by"] = fc.fold.map(cut) - fc.risk
    fc["class"] = np.where(fc.below_cut_by <= NEAR_CUT, "T4 near-cut", "T5 confident")
    fc["cold_start"] = fc.n_history <= 1
    fc["group"] = fc.person_id.map(group)
    fc["frozen_band_refers"] = fc.risk >= th["main"]["low_risk_below"]
    rows = fc[["row", "person_id", "fold", "risk", "below_cut_by", "n_history", "class", "cold_start", "group",
               "frozen_band_refers"]].to_dict("records")  # fmt: skip
    return {"false_clears": len(fc), "by_class": fc["class"].value_counts().to_dict(),
            "cold_start": int(fc.cold_start.sum()),
            "by_fold": {str(k): int(v) for k, v in fc.fold.value_counts().items()},
            "by_group": fc.group.value_counts().to_dict(), "frozen_band_refers": int(fc.frozen_band_refers.sum()),
            "people": int(fc.person_id.nunique()), "meals": rows}  # fmt: skip


def shap(cfg, v2: pd.DataFrame, z: pd.DataFrame, tax: dict) -> dict:
    """E5: exact linear SHAP of the frozen v2 fusion and the intercept-versus-fusion split of served logits."""
    lr = joblib.load(cfg.prototype.sensor_first_bundle)["models"]["sensor"][-1]
    X = z[NAMES].to_numpy()
    coef = lr.coef_[0]
    contrib = (X - X.mean(axis=0)) * coef
    base = float((X @ coef).mean() + lr.intercept_[0])
    t5 = [m for m in tax["meals"] if m["class"] == "T5 confident"]
    local = min(t5 or tax["meals"], key=lambda m: m["risk"])
    i = int(np.where(z.row.to_numpy() == local["row"])[0][0])
    logit_i = float(X[i] @ coef + lr.intercept_[0])
    assert abs(base + contrib[i].sum() - logit_i) < 1e-6, "SHAP terms do not sum to the logit"
    a = v2.dropna(subset=["risk"])
    platt_logit, served_logit = _logit(a.risk_platt), _logit(a.risk)
    intercept = served_logit - platt_logit
    row = v2.set_index("row").loc[local["row"]]
    slope, icpt = json.loads(Path(cfg.prototype.sensor_first_thresholds).read_text())["platt"]
    return {"coefficients": {n: float(c) for n, c in zip(NAMES, coef, strict=True)},
            "fusion_intercept": float(lr.intercept_[0]), "base_logit": base,
            "mean_abs_shap": {n: float(v) for n, v in zip(NAMES, np.abs(contrib).mean(axis=0), strict=True)},
            "local": {"row": int(local["row"]), "frozen_logit": logit_i,
                      "terms": {n: float(v) for n, v in zip(NAMES, contrib[i], strict=True)},
                      "frozen_platt_logit": float(slope * logit_i + icpt),
                      "fold_platt_logit": float(_logit(row.risk_platt)), "served_logit": float(_logit(row.risk)),
                      "intercept_shift": float(_logit(row.risk) - _logit(row.risk_platt)),
                      "n_history": int(row.n_history), "risk": float(row.risk)},
            "split": {"mean_abs_intercept_shift": float(np.abs(intercept).mean()),
                      "mean_abs_centred_fusion_logit": float(np.abs(platt_logit - platt_logit.mean()).mean()),
                      "sd_intercept_shift": float(intercept.std()),
                      "sd_fusion_logit": float(platt_logit.std())}}  # fmt: skip


def heart_rate_sign(v2: pd.DataFrame, z: pd.DataFrame, coef: float) -> dict:
    """E6: is 'higher heart rate, lower risk' between people, within people, or both?"""
    d = v2[["row", "person_id", "spike"]].merge(z[["row", "heart rate"]], on="row")
    per = d.groupby("person_id").agg(hr=("heart rate", "mean"), rate=("spike", "mean"))
    between = float(per.hr.corr(per.rate, method="spearman"))
    d["hr_c"] = d["heart rate"] - d.person_id.map(per.hr)

    def fe(sub: pd.DataFrame) -> float:  # logistic with one intercept per person: the within-person slope
        X = pd.concat([sub[["hr_c"]], pd.get_dummies(sub.person_id, dtype=float)], axis=1).to_numpy()
        return float(LogisticRegression(C=1e4, max_iter=5000, fit_intercept=False).fit(X, sub.spike).coef_[0][0])

    within = fe(d)
    ci = person_bootstrap_ci(d.person_id.to_numpy(), lambda i: fe(d.iloc[i].reset_index(drop=True)), B=300)
    return {"frozen_coefficient": coef, "between_person_spearman": between, "people": int(len(per)),
            "within_person_coefficient": within, "within_person_ci95": [ci[1], ci[2]]}  # fmt: skip


def prediabetes_whatif(v2: pd.DataFrame, group: pd.Series, costs: dict, targets: dict) -> dict:
    """E7, not shipped: withhold low risk for a prediabetic profile until N labelled meals."""
    y = v2.spike.to_numpy().astype(int)
    pre = (v2.person_id.map(group) == "prediabetes").to_numpy()
    out = {}
    for n in (0, 5, 10, 20, None):
        dec = v2.decision.copy()
        hold = pre & (dec == "low risk").to_numpy() & ((v2.n_history < n).to_numpy() if n is not None else True)
        dec[hold] = "can't judge"
        rep = usefulness_report(dec.map(ACTION).to_numpy(), y, costs, targets)
        cl = (dec == "low risk").to_numpy()
        out["never" if n is None else str(n)] = {"cleared": int(cl.sum()), "missed": int(y[cl].sum()),
                                                  "missed_among_cleared": rep["missed_among_cleared"],
                                                  "prediabetes_cleared": int((cl & pre).sum()),
                                                  "prediabetes_missed": int(y[cl & pre].sum()),
                                                  "cost_reduction": rep["cost_reduction"]}  # fmt: skip
    return out


def temporal(v2: pd.DataFrame, z: pd.DataFrame) -> dict:
    """E8: each person's last third of meals by time against the first two thirds."""
    d = v2.merge(z[["row", "meal_time"]], on="row").dropna(subset=["risk"])
    d["rank"] = d.groupby("person_id").meal_time.rank(pct=True)
    out = {}
    for name, part in (("first_two_thirds", d[d["rank"] <= 2 / 3]), ("last_third", d[d["rank"] > 2 / 3])):
        s = part.person_id.to_numpy()
        y, r = part.spike.to_numpy().astype(int), part.risk.to_numpy(float)
        pt, lo, hi = person_bootstrap_ci(s, lambda i, y=y, r=r: float(roc_auc_score(y[i], r[i])))
        cl = part.decision == "low risk"
        out[name] = {"meals": int(len(part)), "auroc": pt, "ci95": [lo, hi], "cleared": int(cl.sum()),
                     "missed_among_cleared": float(part.spike[cl].mean()) if cl.any() else None}  # fmt: skip
    return out


def carb_value() -> dict:
    """E9: the photo model's out-of-fold carbs against typing the training median."""
    o = pd.read_csv("outputs/photo_cv/photo_oof.csv")
    bins = np.array([30, 60, 90])

    def rng(g):
        return np.digitize(np.asarray(g, float), bins)

    truth = rng(o.carbs_g)
    pred_bin = o[["p_bin0", "p_bin1", "p_bin2", "p_bin3"]].to_numpy().argmax(axis=1)
    return {"photos": int(len(o)), "mae_photo_g": float((o.photo_g - o.carbs_g).abs().mean()),
            "mae_median_g": float((o.median_g - o.carbs_g).abs().mean()),
            "range_accuracy_photo": float((pred_bin == truth).mean()),
            "range_accuracy_median": float((rng(o.median_g) == truth).mean())}  # fmt: skip


def run_v2_explain(cfg) -> dict:
    costs, targets = cost_dict(cfg), cfg.targets.model_dump()
    v2 = pd.read_csv(V2 / "dev_answers.csv")
    th = json.loads(Path(cfg.prototype.sensor_first_thresholds).read_text())
    group = pd.read_csv("outputs/cv/oof_predictions.csv").groupby("person_id").group.first()
    z = features(cfg, v2.row.to_numpy())
    a = v2.dropna(subset=["risk"])
    tax = taxonomy(v2, group, th)
    sh = shap(cfg, v2, z, tax)
    res = {"what": __doc__.splitlines()[0], "calibration": calibration(a), "frozen_band_in_sample":
           frozen_band(v2, th, costs, targets), "miss_curve": miss_curve(a), "taxonomy": tax, "shap": sh,
           "heart_rate": heart_rate_sign(v2, z, sh["coefficients"]["heart rate"]),
           "prediabetes_whatif": prediabetes_whatif(v2, group, costs, targets), "temporal": temporal(v2, z),
           "carb_value": carb_value(),
           "cost_shipped": expected_cost(v2.decision.map(ACTION).to_numpy(), v2.spike.to_numpy().astype(int),
                                         costs)}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v2_explain.json").write_text(json.dumps(res, indent=2, default=float))
    c = res["calibration"]
    logger.bind(stage="v2-explain").info("ECE {:.3f} ({:.3f}-{:.3f}), floor q95 {:.3f}: {}; false clears {} {}",
                                         c["observed"], *c["observed_ci95_biased_upward"],
                                         c["perfectly_calibrated_q95"], c["verdict"], tax["false_clears"],
                                         tax["by_class"])  # fmt: skip
    return res
