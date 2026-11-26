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
                         "carbs": float(f.carbs[t]), "kcal": float(f.kcal[t]), "a1c": float(demo.HbA1c[pid]),
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


def matching_checks(M: pd.DataFrame, X: np.ndarray) -> dict:
    """Is the second dataset lined up with the first? (1) Food-log times against the CGM: the mean glucose curve
    around the scored meals should be flat before the logged time and peak after it, for every person.
    (2) Share of each input outside the range v2 was trained on (configs/v2_input_ranges.json).
    (3) Mean heart rate and last glucose beside CGMacros' own."""
    cgm = pd.read_csv(ROOT / "bigideas_cgm.csv", parse_dates=["t"])
    curves = []
    for pid, c in cgm.groupby("id"):
        g = c.set_index("t").glucose.sort_index().groupby(level=0).mean().resample("5min").mean()
        for t in M[(M.person_id == pid) & M.scored].t:
            w = g[t - pd.Timedelta("60min") : t + pd.Timedelta("120min")]
            if len(w) < 30:
                continue
            rel = (np.round((w.index - t).total_seconds() / 300) * 5).astype(int)
            base = g[t - pd.Timedelta("30min") : t].mean()
            curves.append(pd.Series((w - base).to_numpy(), index=rel).groupby(level=0).mean().rename(int(pid)))
    A = pd.DataFrame(curves)
    per = A.groupby(level=0).mean()
    pooled = A.mean()
    rj = json.loads(Path("configs/v2_input_ranges.json").read_text())
    Xs = X[M.scored.to_numpy()]
    outside = {}
    for j, (name, r) in enumerate(rj["ranges"].items()):
        v = Xs[:, j]
        ok = np.isfinite(v)
        outside[name] = float(((v[ok] < r[0]) | (v[ok] > r[1])).mean()) if ok.any() else None
    cg = np.load("data/dataset.npz") if Path("data/dataset.npz").is_file() else None
    return {"peak_minute_after_logged_time_per_person": {int(p): int(per.loc[p].idxmax()) for p in per.index},
            "pooled_mean_change_mgdl": {int(k): float(pooled[k]) for k in (-60, -30, 0, 15, 30, 45, 60, 90, 120)},
            "share_outside_v2_training_range": outside,
            "mean_heart_rate": {"bigideas": float(np.nanmean(Xs[:, 6])),
                                "cgmacros": float(np.nanmean(cg["heart_rate"])) if cg is not None else None},
            "mean_last_glucose": {"bigideas": float(np.nanmean(Xs[:, 0])),
                                  "cgmacros": float(np.nanmean(cg["glucose"][:, -1])) if cg is not None else None}}  # fmt: skip


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
        "matching_checks": matching_checks(M, X),
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


def run_external_part2(cfg) -> dict:
    """Part 2 (docs/external_bigideas.md): does meal content help on free-living meals? Models fitted on CGMacros
    only, with and without logged carbs, scored once on the BIG IDEAs meals of Part 1."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import meal_rows
    from glimpse.training.baselines import logistic
    from glimpse.training.fusion import logged_carbs

    out_file = OUT / "external_part2.json"
    M, X = build(cfg)
    keep = M.scored.to_numpy()
    s, Xs = M[keep].reset_index(drop=True), X[keep]
    bundle = joblib.load("checkpoints/sensor_first/fusion.joblib")
    th = json.loads(Path("checkpoints/sensor_first/thresholds.json").read_text())
    a, b, m = *th["platt"], th["m_personal"]
    ds = MealDataset.load(Path("data/dataset"))
    tr = meal_rows(ds, set(bundle["train_people"]), scored_only=False)
    ytr = ds.labels()[tr]
    base = logistic(bundle["C"], cfg.seed).fit(trend_features(ds, tr), ytr)
    with_c = logistic(bundle["C"], cfg.seed).fit(np.hstack([trend_features(ds, tr), logged_carbs(ds, tr)]), ytr)
    f0 = base.decision_function(Xs)
    drift = float(np.abs(f0 - bundle["models"]["sensor"].decision_function(Xs)).max())
    if drift > 1e-6:
        raise SystemExit(f"the refitted sensor model differs from the frozen v2 by {drift:.2e}; stopped")
    carb = np.log1p(s.carbs.to_numpy(float))[:, None]
    f1 = with_c.decision_function(np.hstack([Xs, carb]))
    y, sid, t = s.spike.to_numpy().astype(int), s.person_id.to_numpy(), s.t.to_numpy("datetime64[s]")

    def with_intercept(f: np.ndarray) -> np.ndarray:
        fp, out = a * f + b, np.zeros(len(f))
        for p in np.unique(sid):
            ix = np.where(sid == p)[0]
            ix = ix[np.argsort(t[ix])]
            out[ix] = fp[ix] + replay_offsets(fp[ix], y[ix].astype(float), t[ix], m)
        return out

    def paired(u: np.ndarray, v: np.ndarray) -> dict:
        def fn(i):
            return float(roc_auc_score(y[i], u[i]) - roc_auc_score(y[i], v[i])) if len(set(y[i])) == 2 else float("nan")

        pt, lo, hi = person_bootstrap_ci(sid, fn)
        return {"with_carbs": float(roc_auc_score(y, u)), "without": float(roc_auc_score(y, v)), "gain": pt, "ci95": [lo, hi]}

    def within(f: np.ndarray) -> float:
        return float(np.mean([roc_auc_score(y[sid == p], f[sid == p]) for p in np.unique(sid) if len(set(y[sid == p])) == 2]))

    ph = pd.read_csv("outputs/photo_cv/photo_oof.csv")
    ph = ph[ph.scored]
    err = (np.log1p(ph.photo_g) - np.log1p(ph.carbs_g)).to_numpy()
    rng, gains = np.random.default_rng(cfg.seed), []
    for _ in range(200):
        noisy = carb[:, 0] + rng.choice(err, len(carb))
        gains.append(float(roc_auc_score(y, with_c.decision_function(np.hstack([Xs, noisy[:, None]]))) - roc_auc_score(y, f0)))
    coef = with_c[-1].coef_[0]
    res = {"what": "External Part 2: meal content on free-living meals (fitted on CGMacros only, scored once on BIG IDEAs)",
           "meals": len(s), "people": int(len(np.unique(sid))), "refit_matches_frozen_v2_max_abs_logit_difference": drift,
           "raw_no_intercept": paired(f1, f0) | {"within_person_with": within(f1), "within_person_without": within(f0)},
           "with_frozen_platt_and_intercept": paired(with_intercept(f1), with_intercept(f0)),
           "photo_sized_error_simulation": {"draws": 200, "log_error_sd": float(err.std()), "mean_gain": float(np.mean(gains)),
                                            "gain_p2.5_p97.5": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]},
           "standardised_carb_coefficient": float(coef[-1]),
           "cgmacros_gain_for_reference": "about +0.03 at most (report section 2: true logged carbs, fusion study)"}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2))
    logger.bind(stage="external").info("{}", json.dumps(res)[:1200])
    return res


def run_external_part3(cfg) -> dict:
    """Part 3 (docs/external_bigideas.md): decisions of the carb-aware model on the external people. The model,
    its Platt map and its cut-offs come from CGMacros only; no per-person intercept."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2, meal_rows
    from glimpse.evaluation.metrics import fit_platt
    from glimpse.training.baselines import logistic
    from glimpse.training.final import cost_dict
    from glimpse.training.fusion import logged_carbs

    M, X = build(cfg)
    keep = M.scored.to_numpy()
    s, Xs = M[keep].reset_index(drop=True), X[keep]
    bundle = joblib.load("checkpoints/sensor_first/fusion.joblib")
    ds, sp = MealDataset.load(Path("data/dataset")), SplitsV2.load(Path("splits/dev_splits_v2.json"))
    yall, costs = ds.labels(), cost_dict(cfg)

    def feats(rows):
        return np.hstack([trend_features(ds, rows), logged_carbs(ds, rows)])

    # out-of-fold scores of the recipe on the 23 development people (CGMacros only)
    f_oof, y_oof, pid_oof = [], [], []
    for k, fold in enumerate(sp.dev_folds):
        tr = meal_rows(ds, set(sp.training_people(k)), scored_only=False)
        te = meal_rows(ds, set(fold), scored_only=True)
        m = logistic(bundle["C"], cfg.seed).fit(feats(tr), yall[tr])
        f_oof.append(m.decision_function(feats(te)))
        y_oof.append(yall[te])
        pid_oof.append([ds.samples[i].person_id for i in te])
    f_oof, y_oof, pid_oof = np.concatenate(f_oof), np.concatenate(y_oof).astype(int), np.concatenate(pid_oof)
    a, b = fit_platt(1 / (1 + np.exp(-f_oof)), y_oof)
    r_oof = 1 / (1 + np.exp(-(a * f_oof + b)))
    tab = np.array([costs["low_risk"], costs["caution"], costs["abstain"]], float)  # rows: clear, caution, refer

    def act(r, lo, hi):
        return np.where(r < lo, 0, np.where(r >= hi, 1, 2))

    def cost(ac, y):
        return float(tab[ac, 1 - y].sum())

    grid = np.round(np.arange(0.05, 0.601, 0.01), 2)
    lo, hi = min(((x, z) for x in grid for z in grid if x <= z), key=lambda c: cost(act(r_oof, *c), y_oof))

    def usefulness(r, y, sid) -> dict:
        ac = act(r, lo, hi)

        def red(i):
            return 1 - cost(ac[i], y[i]) / cost(np.ones(len(i), int), y[i])

        def miss(i):
            c = ac[i] == 0
            return float(y[i][c].mean()) if c.any() else float("nan")

        out = {}
        for name, fn, target, higher in (("cost_reduction", red, 0.10, True),
                                         ("cleared_share", lambda i: float((ac[i] == 0).mean()), 0.20, True),
                                         ("referred_share", lambda i: float((ac[i] == 2).mean()), 0.20, False),
                                         ("missed_among_cleared", miss, 0.10, False)):  # fmt: skip
            pt, l, h = person_bootstrap_ci(sid, fn)
            ok_pt = pt >= target if higher else pt <= target
            ok_all = l >= target if higher else h <= target
            out[name] = {"point": pt, "ci95": [l, h], "target": target,
                         "label": "met" if ok_all else "on track" if ok_pt else "at risk"}  # fmt: skip
        out["cleared"], out["missed"] = int((ac == 0).sum()), int(y[ac == 0].sum())
        return out

    model = logistic(bundle["C"], cfg.seed).fit(feats(meal_rows(ds, set(bundle["train_people"]), scored_only=False)),
                                                yall[meal_rows(ds, set(bundle["train_people"]), scored_only=False)])  # fmt: skip
    f = model.decision_function(np.hstack([Xs, np.log1p(s.carbs.to_numpy(float))[:, None]]))
    risk = 1 / (1 + np.exp(-(a * f + b)))
    y, sid = s.spike.to_numpy().astype(int), s.person_id.to_numpy()
    ext = usefulness(risk, y, sid)
    res = {"what": "External Part 3: decisions of the carb-aware model (model, Platt map and cut-offs from CGMacros only)",
           "platt_from_cgmacros": [a, b], "cutoffs_from_cgmacros": [float(lo), float(hi)],
           "cgmacros_out_of_fold": {"meals": len(y_oof), "auroc": float(roc_auc_score(y_oof, r_oof)),
                                    "usefulness": usefulness(r_oof, y_oof, pid_oof)},
           "external": {"meals": len(y), "people": int(len(np.unique(sid))), "auroc": float(roc_auc_score(y, risk)),
                        "brier": float(np.mean((risk - y) ** 2)), "brier_constant": float(y.mean() * (1 - y.mean())),
                        "ece": _ece(y, risk), "mean_risk_minus_spike_rate": float(risk.mean() - y.mean()),
                        "usefulness": ext,
                        "targets_met_or_on_track": int(sum(v["label"] != "at risk" for v in ext.values() if isinstance(v, dict)))}}  # fmt: skip
    (OUT / "external_part3.json").write_text(json.dumps(res, indent=2))
    logger.bind(stage="external").info("{}", json.dumps(res)[:1500])
    return res


def run_external_checks(cfg) -> dict:
    """Follow-up checks a grader asked for (27 Nov, after Parts 1 to 3): is the intercept's effect on ranking
    distinguishable from zero; is the carb gain spread over people; does it survive without snacks (entries
    under 200 kcal removed); how many people does frozen v2 rank below chance."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import meal_rows
    from glimpse.training.baselines import logistic
    from glimpse.training.fusion import logged_carbs

    M, X = build(cfg)
    keep = M.scored.to_numpy()
    s, Xs = M[keep].reset_index(drop=True), X[keep]
    ans = pd.read_csv(OUT / "external_answers.csv")
    y, sid = s.spike.to_numpy().astype(int), s.person_id.to_numpy()
    assert (ans.spike.to_numpy() == y).all()
    r_int, r_raw = ans.risk.to_numpy(), ans.risk_platt_only.to_numpy()

    def paired(u, v, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask

        def fn(i):
            yy = y[m][i]
            return float(roc_auc_score(yy, u[m][i]) - roc_auc_score(yy, v[m][i])) if len(set(yy)) == 2 else float("nan")

        pt, lo, hi = person_bootstrap_ci(sid[m], fn)
        return {"point": pt, "ci95": [lo, hi], "meals": int(m.sum())}

    bundle = joblib.load("checkpoints/sensor_first/fusion.joblib")
    ds = MealDataset.load(Path("data/dataset"))
    tr = meal_rows(ds, set(bundle["train_people"]), scored_only=False)
    with_c = logistic(bundle["C"], cfg.seed).fit(np.hstack([trend_features(ds, tr), logged_carbs(ds, tr)]), ds.labels()[tr])
    f1 = with_c.decision_function(np.hstack([Xs, np.log1p(s.carbs.to_numpy(float))[:, None]]))
    f0 = bundle["models"]["sensor"].decision_function(Xs)
    per = {}
    for p in np.unique(sid):
        k = sid == p
        if len(set(y[k])) == 2:
            per[int(p)] = {"v2_frozen": float(roc_auc_score(y[k], r_int[k])), "without_carbs": float(roc_auc_score(y[k], f0[k])),
                           "with_carbs": float(roc_auc_score(y[k], f1[k]))}
    big = s.kcal.to_numpy(float) >= 200
    res = {"what": "External follow-up checks (27 Nov), same 201 meals of 12 people",
           "intercept_minus_no_intercept_auroc": paired(r_int, r_raw),
           "people_with_both_outcomes": len(per),
           "people_ranked_below_chance_by_frozen_v2": int(sum(v["v2_frozen"] < 0.5 for v in per.values())),
           "people_ranked_better_with_carbs": int(sum(v["with_carbs"] > v["without_carbs"] for v in per.values())),
           "carb_gain_all_meals": paired(f1, f0),
           "carb_gain_meals_of_200_kcal_or_more": paired(f1, f0, big) | {"spike_rate": float(y[big].mean())},
           "carbs_alone_meals_of_200_kcal_or_more": float(roc_auc_score(y[big], s.carbs.to_numpy(float)[big])),
           "per_person_within_auroc": per}  # fmt: skip
    (OUT / "external_checks.json").write_text(json.dumps(res, indent=2))
    return res
