# ruff: noqa: E501, E741
"""External check on a second dataset, 27 Nov (docs/external_bigideas.md): the frozen sensor-first model v2 on
the 16 people of the BIG IDEAs glycemic-wearable dataset (PhysioNet). Exploratory; the served v1 needs a photo
and is not scored (`glimpse external-bigideas`, needs data/bigideas/)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

from glimpse.calibration import replay_offsets
from glimpse.data.preprocess import passes_premeal_checks, resample_window
from glimpse.evaluation.metrics import person_bootstrap_ci
from glimpse.training.baselines import trend_features

ROOT, OUT = Path("data/bigideas"), Path("outputs/external_bigideas")
DEV_SPIKE_RATE = 0.328  # CGMacros development spike rate (report section 3), the running-rate baseline's anchor


def build(cfg) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per usable meal with the label and flags, and the 15 inputs v2 reads (missing ones as NaN)."""
    L, W, pc = cfg.label, cfg.window_rules, cfg.pipeline
    cgm = pd.read_csv(ROOT / "bigideas_cgm.csv", parse_dates=["t"])
    hr = pd.read_csv(ROOT / "bigideas_hr5.csv", parse_dates=["t"])
    food = pd.read_csv(ROOT / "bigideas_food.csv")
    food["t"] = pd.to_datetime(food.time_begin, errors="coerce")
    demo = pd.read_csv(ROOT / "Demographics.csv", encoding="utf-8-sig").set_index("ID")
    horizon, base_w = pd.Timedelta(minutes=L.horizon_min), pd.Timedelta(minutes=L.baseline_window_min)
    rows, glu, hrs = [], [], []
    for pid, c in cgm.groupby("id"):
        raw = c.set_index("t").glucose.sort_index()
        g = raw.groupby(level=0).mean().resample("5min").mean()
        h = hr[hr.id == pid].set_index("t").hr.sort_index()
        f = (
            food[(food.id == pid) & food.t.notna()]
            .groupby("t")
            .agg(kcal=("calorie", "sum"), carbs=("total_carb", "sum"))
        )
        times = list(f[f.kcal > 0].index)
        for i, t in enumerate(times):
            pre = g[t - base_w : t].dropna()
            post = g[t + pd.Timedelta("1min") : t + horizon]
            obs = post.dropna()
            if len(pre) < W.min_baseline_readings or len(obs) == 0:
                continue
            edges = [t, *obs.index, t + horizon]
            gap = max((b - a).total_seconds() / 60 for a, b in zip(edges[:-1], edges[1:], strict=False))
            if not (post.notna().mean() >= W.min_post_coverage and gap <= W.max_gap_min):
                continue
            win, hwin = (
                raw[t - pd.Timedelta(minutes=pc.cgm_lookback_min) : t],
                h[t - pd.Timedelta(minutes=pc.cgm_lookback_min) : t],
            )
            gw = resample_window(
                list(win.index.to_pydatetime()), list(win.values), t.to_pydatetime(), pc.cgm_lookback_min
            )
            hw = resample_window(
                list(hwin.index.to_pydatetime()), list(hwin.values), t.to_pydatetime(), pc.cgm_lookback_min
            )
            rise = float(obs.max() - pre.mean())
            rows.append({"person_id": int(pid), "t": t, "rise": rise, "spike": int(rise >= L.spike_threshold_mgdl),
                         "carbs": float(f.carbs[t]), "a1c": float(demo.HbA1c[pid]),
                         "next_meal_within": any(t < u <= t + horizon for u in times[i + 1 : i + 3]),
                         "premeal_ok": bool(passes_premeal_checks(gw, pc.min_cgm_readings_last_30min,
                                                                  pc.min_cgm_coverage, pc.max_cgm_gap_min))})  # fmt: skip
            glu.append(gw.values)
            hrs.append(hw.values)
    M = pd.DataFrame(rows)
    start = M.groupby("person_id").t.transform("min")
    on_h = pd.Timedelta(hours=cfg.onboarding.hours)
    M["onboarding"] = (M.t + horizon) <= start + on_h
    M["scored"] = (M.t >= start + on_h) & ~M.next_meal_within & M.premeal_ok
    on = M[M.onboarding].groupby("person_id").agg(n=("spike", "size"), rate=("spike", "mean"), prise=("rise", "mean"))
    on.loc[on.n < cfg.onboarding.min_meals_for_personal_prior, ["rate", "prise"]] = np.nan
    M = M.join(on[["rate", "prise"]], on="person_id")
    hour = M.t.dt.hour + M.t.dt.minute / 60
    prof = np.column_stack([np.full(len(M), np.nan), np.full(len(M), np.nan), M.a1c, np.full(len(M), np.nan),
                            np.sin(2 * math.pi * hour / 24), np.cos(2 * math.pi * hour / 24), M.rate, M.prise])  # fmt: skip
    fake = SimpleNamespace(glucose=np.array(glu, float), heart_rate=np.array(hrs, float), profile=prof)
    keep = M.premeal_ok.to_numpy()  # trend features need a readable window
    X = np.full((len(M), 15), np.nan)
    X[keep] = trend_features(fake, np.where(keep)[0])
    return M, X


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in np.unique(b)))


def run_external_bigideas(cfg) -> dict:
    from glimpse.training.final import cost_dict

    log = logger.bind(stage="external")
    M, X = build(cfg)
    bundle = joblib.load("checkpoints/sensor_first/fusion.joblib")
    th = json.loads(Path("checkpoints/sensor_first/thresholds.json").read_text())
    a, b = th["platt"]
    lo, hi, m = th["main"]["low_risk_below"], th["main"]["caution_from"], th["m_personal"]
    s = M[M.scored].copy()
    f = a * bundle["models"]["sensor"].decision_function(X[M.scored.to_numpy()]) + b
    s["logit_platt"] = f
    risk, rate = np.zeros(len(s)), np.zeros(len(s))
    for _pid, ix in s.groupby("person_id").indices.items():
        g = s.iloc[ix].sort_values("t")
        pos = [s.index.get_loc(i) for i in g.index]
        y, t = g.spike.to_numpy(float), g.t.to_numpy("datetime64[s]")
        off = replay_offsets(g.logit_platt.to_numpy(), y, t, m)
        risk[pos] = 1 / (1 + np.exp(-(g.logit_platt.to_numpy() + off)))
        lag = np.timedelta64(120 * 60, "s")
        for j, p in enumerate(pos):
            hist = t + lag <= t[j]
            rate[p] = (y[hist].sum() + m * DEV_SPIKE_RATE) / (hist.sum() + m)
    s["risk"], s["risk_platt_only"], s["running_rate"] = risk, 1 / (1 + np.exp(-f)), rate
    y, sid = s.spike.to_numpy().astype(int), s.person_id.to_numpy()

    def auc(col: str) -> dict:
        v = s[col].to_numpy(float)
        ok = np.isfinite(v)
        pt, l, h = person_bootstrap_ci(sid[ok], lambda i: float(roc_auc_score(y[ok][i], v[ok][i]))
                                       if len(set(y[ok][i])) == 2 else float("nan"))  # fmt: skip
        return {"auroc": pt, "ci95": [l, h], "meals": int(ok.sum())}

    def within(col: str) -> dict:
        v = [roc_auc_score(g.spike, g[col]) for _, g in s.groupby("person_id") if g.spike.nunique() == 2]
        return {"mean": float(np.mean(v)), "people": len(v)}

    dec = np.where(risk < lo, "low risk", np.where(risk >= hi, "caution", "can't judge"))
    costs = cost_dict(cfg)
    tab = {"low risk": costs["low_risk"], "caution": costs["caution"], "can't judge": costs["abstain"]}

    def cost(d, yy):
        return float(sum(tab[k][0] if v else tab[k][1] for k, v in zip(d, yy, strict=True)))

    def red(i):
        return 1 - cost(dec[i], y[i]) / cost(np.full(len(i), "caution"), y[i])

    cr = person_bootstrap_ci(sid, red)
    cl = dec == "low risk"
    res = {
        "what": __doc__.splitlines()[0], "plan": "docs/external_bigideas.md",
        "people": int(s.person_id.nunique()), "meals_usable": len(M), "meals_scored": len(s),
        "spike_rate": float(y.mean()), "people_with_onboarding_priors": int(s.groupby("person_id").rate.first().notna().sum()),
        "heart_rate_missing_share": float(np.isnan(X[M.scored.to_numpy()][:, 6]).mean()),
        "v2_frozen": auc("risk") | {"within_person": within("risk"), "brier": float(np.mean((risk - y) ** 2)),
                                    "brier_constant": float(y.mean() * (1 - y.mean())), "ece": _ece(y, risk),
                                    "mean_risk_minus_spike_rate": float(risk.mean() - y.mean())},
        "v2_without_intercept": auc("risk_platt_only") | {"within_person": within("risk_platt_only")},
        "running_spike_rate": auc("running_rate"),
        "logged_carbs_alone": auc("carbs") | {"within_person": within("carbs")},
        "decisions_at_frozen_cutoffs": {
            "cutoffs": [lo, hi], "cleared_share": float(cl.mean()), "cleared": int(cl.sum()), "missed": int(y[cl].sum()),
            "missed_among_cleared": float(y[cl].mean()) if cl.any() else None,
            "referred_share": float((dec == "can't judge").mean()),
            "cost_reduction_vs_always_caution": cr[0], "cost_reduction_ci95": [cr[1], cr[2]]},
        "per_person": [{"person_id": int(p), "meals": len(g), "spikes": int(g.spike.sum()),
                        "auroc": float(roc_auc_score(g.spike, g.risk)) if g.spike.nunique() == 2 else None}
                       for p, g in s.groupby("person_id")],
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "external_bigideas.json").write_text(json.dumps(res, indent=2))
    s[["person_id", "t", "spike", "rise", "risk", "risk_platt_only", "running_rate"]].assign(decision=dec).to_csv(
        OUT / "external_answers.csv", index=False)  # fmt: skip
    log.info("{}", json.dumps({k: v for k, v in res.items() if k != "per_person"})[:1500])
    return res
