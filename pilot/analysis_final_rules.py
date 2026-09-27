"""Re-check pilot under the FINAL proposal rules: usable window = >=3 baseline readings, >=70% post readings, no gap > 20 min;
share < 10% excluded; 72-h onboarding; plus fully-consumed-only sensitivity. Same GBM + person-grouped 5-fold x 3 seeds."""
import numpy as np, pandas as pd, glob, re, os
src = open("analysis2.py").read()
exec(src.split("M = M.sort_values")[0].replace("print(", "(lambda *a, **k: None)("))
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
# recompute window quality with Dexcom 5-min grid
G = {}
for f in sorted(glob.glob(f"{R}/CGMacros-*/CGMacros-*.csv")):
    sid = int(re.findall(r"(\d+)", os.path.basename(f))[-1]); d = pd.read_csv(f); d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    G[sid] = pd.to_numeric(d.set_index("Timestamp")["Dexcom GL"], errors="coerce").dropna().groupby(level=0).mean().resample("5min").mean()
def quality(r):
    g = G[r.sid]; pre = g[r.t - pd.Timedelta("30min"):r.t].dropna(); post = g[r.t + pd.Timedelta("1min"):r.t + pd.Timedelta("120min")]
    obs = post.dropna().index
    if len(obs) == 0: return pd.Series(dict(nbase=len(pre), maxgap=999))
    edges = [r.t] + list(obs) + [r.t + pd.Timedelta("120min")]
    gap = max((b - a).total_seconds() / 60 for a, b in zip(edges[:-1], edges[1:]))
    return pd.Series(dict(nbase=len(pre), maxgap=gap))
M = M.join(M.apply(quality, axis=1))
M = M.sort_values(["sid", "t"]); start = M.groupby("sid").t.transform("min")
M["s50"] = (M.rise >= 50).astype(int)
sh = M.eaten.values.astype(float); sh = np.where(sh <= 1.0, sh * 100, sh); sh = np.where((sh > 100) | (sh <= 0) | np.isnan(sh), 100, sh); M["share"] = sh
n0 = len(M)
M = M[(M.nbase >= 3) & (M.maxgap <= 20) & (M.share >= 10)].copy()
print(f"meals with 70% window: {n0} -> after baseline>=3, max gap<=20 min, share>=10%: {len(M)}")
M["onb"] = (M.t + pd.Timedelta("2h")) <= start.loc[M.index] + pd.Timedelta("72h"); M["after"] = M.t >= start.loc[M.index] + pd.Timedelta("72h")
on = M[M.onb]; n_on = on.groupby("sid").size()
M = M.join(on.groupby("sid").s50.mean().rename("prior_rate"), on="sid").join(on.groupby("sid").rise.mean().rename("prior_rise"), on="sid")
for c in ["carbs", "protein", "fat", "fiber", "kcal"]: M[c + "_served"] = M[c] * 100 / M["share"]
E = M[M.after & ~M.overlap & M.rise.notna()].copy()
print("onboarding", int(M.onb.sum()), "| crosses boundary", int((~M.onb & ~M.after).sum()), "| after", int(M.after.sum()), "| evaluated", len(E), "| people with <5 onboarding meals", int((n_on < 5).sum()) + (45 - len(n_on)))
MAC = ["carbs", "protein", "fat", "fiber", "kcal"]; LAB = ["a1c", "fglu", "fins", "BMI", "Age", "Triglycerides"]; TRAJ = ["slope60", "slope20", "sd120", "min120", "since_meal", "hr_slope", "mets60"]
base = ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB + TRAJ
def folds(sids, k=5, seed=0):
    u = np.array(sorted(set(sids))); r = np.random.default_rng(seed); r.shuffle(u); return [np.isin(sids, u[i::k]) for i in range(k)]
def oof(df, fs):
    P = np.zeros((3, len(df)))
    for si in range(3):
        for te in folds(df.sid.values, 5, si):
            P[si, te] = HGB(max_depth=3, learning_rate=.05, max_iter=200).fit(df[fs][~te], df.s50[~te]).predict_proba(df[fs][te])[:, 1]
    return P.mean(0)
for name, df in [("all 45", E), ("target cohort", E[E.group.isin(["healthy", "prediabetes"])]), ("target cohort, fully consumed only", E[E.group.isin(["healthy", "prediabetes"]) & (E.share >= 100)])]:
    df = df.reset_index(drop=True)
    print(f"{name}: n={len(df)}, people={df.sid.nunique()}, spike rate={df.s50.mean():.3f}")
    for lab, mac in [("consumed", MAC), ("served", [c + "_served" for c in MAC])]:
        print(f"   {lab:9s} macros AUROC {roc_auc_score(df.s50, oof(df, mac + base)):.3f}")
