"""Pilot rerun with review fixes: 72-h time-based onboarding (onboarding windows must end before the boundary),
served macros (= consumed / share eaten), and results for the target cohort (healthy + prediabetes)."""
import numpy as np, pandas as pd
src = open("analysis2.py").read()
exec(src.split("M = M.sort_values")[0].replace("print(", "(lambda *a, **k: None)("))
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
M = M.sort_values(["sid", "t"]); start = M.groupby("sid").t.transform("min")
M["onb"] = (M.t + pd.Timedelta("2h")) <= start + pd.Timedelta("72h"); M["after"] = M.t >= start + pd.Timedelta("72h")
M["s50"] = (M.rise >= 50).astype(int)
on = M[M.onb]; n_on = on.groupby("sid").size()
print("onboarding meals per person: min", n_on.min(), "median", n_on.median(), "people with <5:", (n_on < 5).sum())
M = M.join(on.groupby("sid").s50.mean().rename("prior_rate"), on="sid").join(on.groupby("sid").rise.mean().rename("prior_rise"), on="sid")
sh = M.eaten.values.astype(float); sh = np.where(sh <= 1.0, sh * 100, sh); sh = np.where((sh > 100) | (sh <= 0) | np.isnan(sh), 100, sh); M["share"] = sh
for c in ["carbs", "protein", "fat", "fiber", "kcal"]: M[c + "_served"] = M[c] * 100 / M["share"]
print("meals with share < 100%:", round((M.share < 100).mean(), 3))
E = M[M.after & ~M.overlap & M.rise.notna()].copy()
print("counts: total", len(M), "| onboarding", M.onb.sum(), "| after 72h", M.after.sum(), "| after & no overlap", len(E), "| window crosses boundary", (~M.onb & ~M.after).sum())
MAC = ["carbs", "protein", "fat", "fiber", "kcal"]; LAB = ["a1c", "fglu", "fins", "BMI", "Age", "Triglycerides"]; TRAJ = ["slope60", "slope20", "sd120", "min120", "since_meal", "hr_slope", "mets60"]
def folds(sids, k=5, seed=0):
    u = np.array(sorted(set(sids))); r = np.random.default_rng(seed); r.shuffle(u); return [np.isin(sids, u[i::k]) for i in range(k)]
def oof(df, fs, seeds=(0, 1, 2)):
    P = np.zeros((len(seeds), len(df)))
    for si, s in enumerate(seeds):
        for te in folds(df.sid.values, 5, s):
            m = HGB(max_depth=3, learning_rate=.05, max_iter=200).fit(df[fs][~te], df.s50[~te]); P[si, te] = m.predict_proba(df[fs][te])[:, 1]
    return P.mean(0)
base = ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB + TRAJ
for name, df in [("all 45", E), ("target cohort (healthy+prediabetes)", E[E.group.isin(["healthy", "prediabetes"])])]:
    df = df.reset_index(drop=True)
    print(f"{name}: n={len(df)}, people={df.sid.nunique()}, spike rate={df.s50.mean():.3f}")
    for lab, mac in [("consumed macros", MAC), ("served macros", [c + "_served" for c in MAC])]:
        p = oof(df, mac + base); print(f"   {lab:16s} GBM+trend AUROC {roc_auc_score(df.s50, p):.3f}")
