"""Meal-level dataset construction from CGMacros under the proposal's prespecified rules.

Only information available BEFORE eating is used as model input. Consumed macros and share eaten are used
solely to reconstruct the served-macro reference (consumed / share, proportional-consumption assumption).
"""
from __future__ import annotations
import glob, os, re
import numpy as np
import pandas as pd

MACROS = ["carbs", "protein", "fat", "fiber", "kcal"]


def _slope(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 3:
        return np.nan
    x = (s.index - s.index[0]).total_seconds() / 60
    return float(np.polyfit(x, s.values, 1)[0])


def _share_pct(v) -> float:
    """Normalise the inconsistently coded 'Amount Consumed' field to a percentage (NaN/0/>100 -> 100)."""
    v = pd.to_numeric(v, errors="coerce")
    if pd.isna(v) or v <= 0 or v > 100:
        return 100.0
    return float(v * 100) if v <= 1 else float(v)


def load_bio(root: str) -> pd.DataFrame:
    b = pd.read_csv(os.path.join(root, "bio.csv"))
    b.columns = [c.strip() for c in b.columns]
    b = b.rename(columns={"subject": "sid", "A1c PDL (Lab)": "a1c", "Fasting GLU - PDL (Lab)": "fglu", "Insulin": "fins"})
    b["group"] = pd.cut(b.a1c, [0, 5.69, 6.49, 99], labels=["healthy", "prediabetes", "T2D"]).astype(str)
    return b[["sid", "a1c", "fglu", "fins", "BMI", "Age", "Triglycerides", "group"]]


def build_meals(root: str, cfg: dict) -> pd.DataFrame:
    """Return one row per labelled meal with pre-meal features, served/consumed macros, label and quality flags."""
    L, W, Mc = cfg["label"], cfg["window_rules"], cfg["meals"]
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "CGMacros-*", "CGMacros-*.csv"))):
        sid = int(re.findall(r"(\d+)", os.path.basename(f))[-1])
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d["Timestamp"] = pd.to_datetime(d["Timestamp"])
        d = d.set_index("Timestamp").sort_index()
        g = pd.to_numeric(d["Dexcom GL"], errors="coerce").dropna().groupby(level=0).mean().resample("5min").mean()
        hr = pd.to_numeric(d.get("HR"), errors="coerce") if "HR" in d else pd.Series(np.nan, index=d.index)
        mets = pd.to_numeric(d.get("METs"), errors="coerce") if "METs" in d else pd.Series(np.nan, index=d.index)
        meals = d[pd.to_numeric(d["Calories"], errors="coerce").fillna(0) > 0]
        times = list(meals.index)
        for i, t in enumerate(times):
            pre = g[t - pd.Timedelta(minutes=L["baseline_window_min"]):t].dropna()
            post = g[t + pd.Timedelta("1min"):t + pd.Timedelta(minutes=L["horizon_min"])]
            obs = post.dropna()
            if len(pre) < W["min_baseline_readings"] or len(obs) == 0:
                continue
            edges = [t] + list(obs.index) + [t + pd.Timedelta(minutes=L["horizon_min"])]
            max_gap = max((b - a).total_seconds() / 60 for a, b in zip(edges[:-1], edges[1:]))
            coverage = post.notna().mean()
            r = meals.loc[t]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            share = _share_pct(r.get("Amount Consumed"))
            prev = [u for u in times[:i] if u < t]
            row = dict(
                sid=sid, t=t, base=pre.mean(), rise=obs.max() - pre.mean(), coverage=coverage, max_gap=max_gap,
                usable=(coverage >= W["min_post_coverage"]) and (max_gap <= W["max_gap_min"]),
                usable_strict=(coverage >= W["strict"]["min_post_coverage"]) and (max_gap <= W["strict"]["max_gap_min"]),
                next_meal_within=any(t < u <= t + pd.Timedelta(minutes=Mc["exclude_next_meal_within_min"]) for u in times[i + 1:i + 3]),
                share=share, photo=r.get("Image path"),
                slope60=_slope(g[t - pd.Timedelta("60min"):t]), slope20=_slope(g[t - pd.Timedelta("20min"):t]),
                sd120=g[t - pd.Timedelta("120min"):t].std(), min120=g[t - pd.Timedelta("120min"):t].min(),
                since_meal=(t - prev[-1]).total_seconds() / 3600 if prev else 12.0,
                hr30=hr[t - pd.Timedelta("30min"):t].mean(), hr_slope=_slope(hr[t - pd.Timedelta("60min"):t]),
                mets60=mets[t - pd.Timedelta("60min"):t].mean(), hour=t.hour + t.minute / 60,
            )
            for m in MACROS:
                col = {"carbs": "Carbs", "protein": "Protein", "fat": "Fat", "fiber": "Fiber", "kcal": "Calories"}[m]
                row[f"{m}_consumed"] = pd.to_numeric(r.get(col), errors="coerce")
                row[f"{m}_served"] = row[f"{m}_consumed"] * 100.0 / share
            rows.append(row)
    M = pd.DataFrame(rows).merge(load_bio(root), on="sid", how="left")
    for thr in [L["spike_threshold_mgdl"]] + list(L["sensitivity_thresholds"]):
        M[f"spike{thr}"] = (M.rise >= thr).astype(int)
    M = M[M.share >= Mc["min_share_eaten_pct"]]
    # 72-h onboarding, measured from each person's first labelled meal
    start = M.groupby("sid").t.transform("min")
    h = pd.Timedelta(hours=cfg["onboarding"]["hours"])
    M["onboarding"] = (M.t + pd.Timedelta(minutes=L["horizon_min"])) <= start + h
    M["post_onboarding"] = M.t >= start + h
    return M.sort_values(["sid", "t"]).reset_index(drop=True)


def add_personal_prior(M: pd.DataFrame, cfg: dict, train_target_sids=None) -> pd.DataFrame:
    """Personal prior from onboarding meals; population prior (training target cohort only) if < min meals."""
    thr = cfg["label"]["spike_threshold_mgdl"]
    on = M[M.onboarding & M.usable]
    stats = on.groupby("sid").agg(n=(f"spike{thr}", "size"), prior_rate=(f"spike{thr}", "mean"), prior_rise=("rise", "mean"))
    pool = on if train_target_sids is None else on[on.sid.isin(train_target_sids)]
    pop_rate, pop_rise = pool[f"spike{thr}"].mean(), pool.rise.mean()
    M = M.join(stats, on="sid")
    few = M.n.fillna(0) < cfg["onboarding"]["min_meals_for_personal_prior"]
    M.loc[few, "prior_rate"], M.loc[few, "prior_rise"] = pop_rate, pop_rise
    return M.drop(columns="n")


def evaluation_set(M: pd.DataFrame, strict: bool = False, include_overlap: bool = False) -> pd.DataFrame:
    ok = M.post_onboarding & (M.usable_strict if strict else M.usable) & M.rise.notna()
    if not include_overlap:
        ok &= ~M.next_meal_within
    return M[ok].reset_index(drop=True)
