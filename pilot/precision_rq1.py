"""How precise can RQ1 be? Simulate photo-predicted macros as logged served macros x lognormal error (CV 25% / 40%),
compute OOF GBM predictions (people held out), then the paired person-bootstrap 95% CI of
Delta = AUROC(photo) - AUROC(logged) on (a) all 31 target people, (b) random 8-person locked-set-sized subsets."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("analysis_final_rules.py").read().split('MAC = ["carbs"')[0].replace("print(", "(lambda *a, **k: None)("))
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
E = E.reset_index(drop=True)
MACS = ["carbs_served", "protein_served", "fat_served", "fiber_served", "kcal_served"]
LAB = ["a1c", "fglu", "fins", "BMI", "Age", "Triglycerides"]; TRAJ = ["slope60", "slope20", "sd120", "min120", "since_meal", "hr_slope", "mets60"]
REST = ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB + TRAJ
y = E.s50.values; sids = E.sid.values; tgt = E.group.isin(["healthy", "prediabetes"]).values
def oof(X):
    P = np.zeros((3, len(E)))
    for s in range(3):
        u = np.array(sorted(set(sids))); np.random.default_rng(s).shuffle(u)
        for i in range(5):
            te = np.isin(sids, u[i::5]); P[s, te] = HGB(max_depth=3, learning_rate=.05, max_iter=200).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    return P.mean(0)
p_log = oof(E[MACS + REST].values)
rng = np.random.default_rng(1)
yt, st = y[tgt], sids[tgt]; ppl = np.unique(st); idx = {p: np.where(st == p)[0] for p in ppl}
def ci(pa, pb, people, B=800):
    d = []
    for _ in range(B):
        pick = np.concatenate([idx[p] for p in rng.choice(people, len(people))])
        if yt[pick].min() != yt[pick].max(): d.append(roc_auc_score(yt[pick], pa[pick]) - roc_auc_score(yt[pick], pb[pick]))
    return np.percentile(d, [2.5, 97.5])
for cv in [0.25, 0.40]:
    X = E[MACS].values * rng.lognormal(0, cv, (len(E), 1)) ; p_ph = oof(np.column_stack([X, E[REST].values]))
    a, b = p_ph[tgt], p_log[tgt]
    delta = roc_auc_score(yt, a) - roc_auc_score(yt, b); lo, hi = ci(a, b, ppl)
    print(f"macro error {int(cv*100)}%: Delta (31 target people) {delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] -> {'non-inferior' if lo > -0.03 else ('inferior' if hi < -0.03 else 'inconclusive')}")
    res = []
    for _ in range(60):
        sub = rng.choice(ppl, 8, replace=False); m = np.isin(st, sub)
        if yt[m].min() == yt[m].max(): continue
        d = roc_auc_score(yt[m], a[m]) - roc_auc_score(yt[m], b[m])
        l, h = ci(a, b, sub, 200); res.append((d, l, h))
    r = np.array(res)
    print(f"   8-person subsets: median CI width {np.median(r[:,2]-r[:,1]):.3f}; non-inferior in {np.mean(r[:,1] > -0.03):.0%}, inferior {np.mean(r[:,2] < -0.03):.0%}, inconclusive {np.mean((r[:,1] <= -0.03) & (r[:,2] >= -0.03)):.0%}")
