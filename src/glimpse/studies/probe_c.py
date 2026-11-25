"""Probe C: profile and onboarding priors counterfactual (registered 26 Nov in docs/probes_s13.md before
its run; `glimpse probe-c`). Data-free: the frozen fusion on the committed standardised inputs of the 460
answered development meals, Platt only, deployed cut-offs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

OUT = Path("outputs/probes")
INPUTS = ["HbA1c", "fasting glucose", "prior spike rate", "prior rise"]


def run_probe_c(cfg) -> dict:
    """Write outputs/probes/probe_c.json."""
    import joblib

    from glimpse.serve.rules import decision
    from glimpse.training.final import FINAL, load_thresholds

    art = joblib.load(FINAL)
    pipe = art["models"]["photo"]
    imp, sc, lr = pipe.named_steps.values()
    a, b = art["calibration"]
    th = load_thresholds()["main"]
    lo, hi = th["low_risk_below"], th["caution_from"]
    z = pd.read_csv("outputs/figures/shap_features_z.csv").set_index("row")
    w, c = lr.coef_[0], float(lr.intercept_[0])

    def risk(zz: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(a * (zz @ w + c) + b)))

    def decide(r: np.ndarray) -> np.ndarray:
        return np.array([decision(float(x), lo, hi) for x in r])

    base = z.to_numpy()
    res = {"what": __doc__.splitlines()[0], "n_meals": len(z), "cutoffs": [lo, hi], "inputs": {}}
    for name in INPUTS:
        j = list(z.columns).index(name)
        p10, p90 = np.quantile(base[:, j], [0.1, 0.9])
        lo_z, hi_z = base.copy(), base.copy()
        lo_z[:, j], hi_z[:, j] = p10, p90
        r10, r90 = risk(lo_z), risk(hi_z)
        res["inputs"][name] = {"coefficient_per_sd": float(w[j]),
                               "share_risk_rises_10th_to_90th": float(np.mean(r90 > r10)),
                               "median_risk_change": float(np.median(r90 - r10)),
                               "share_decision_differs": float(np.mean(decide(r10) != decide(r90))),
                               "expected_direction_met": bool(np.mean(r90 > r10) >= 0.9)}  # fmt: skip
    idx = [list(z.columns).index(n) for n in ("prior spike rate", "prior rise")]
    med = base.copy()
    med[:, idx] = (imp.statistics_[idx] - sc.mean_[idx]) / sc.scale_[idx]
    r0, rm = risk(base), risk(med)
    d0, dm = decide(r0), decide(rm)
    res["imputed_priors"] = {"medians": imp.statistics_[idx].tolist(),
                             "median_abs_risk_change": float(np.median(np.abs(rm - r0))),
                             "share_decision_differs": float(np.mean(d0 != dm)),
                             "cleared_own_priors": int((d0 == "low risk").sum()),
                             "cleared_imputed": int((dm == "low risk").sum())}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_c.json").write_text(json.dumps(res, indent=2))
    for k, v in res["inputs"].items():
        logger.bind(stage="probe-c").info("{}: rises on {:.0%}, median change {:+.3f}, decisions differ {:.0%}",
                                          k, v["share_risk_rises_10th_to_90th"], v["median_risk_change"],
                                          v["share_decision_differs"])  # fmt: skip
    return res
