"""RQ1 anchor: how much do meal macros add over a no-meal-information model (sensor + profile + prior)?
Final rules, target cohort, GBM, 5-fold person CV x 3 seeds, paired person bootstrap."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
exec(open("analysis_final_rules.py").read().split('MAC = ["carbs"')[0].replace("print(", "(lambda *a, **k: None)("))
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
E = E.reset_index(drop=True)
SERVED = ["carbs_served", "protein_served", "fat_served", "fiber_served", "kcal_served"]
LAB = ["a1c", "fglu", "fins", "BMI", "Age", "Triglycerides"]; TRAJ = ["slope60", "slope20", "sd120", "min120", "since_meal", "hr_slope", "mets60"]
REST = ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB + TRAJ
y = E.s50.values; sids = E.sid.values; tgt = E.group.isin(["healthy", "prediabetes"]).values
def oof(cols):
    P = np.zeros((3, len(E)))
    for s in range(3):
        u = np.array(sorted(set(sids))); np.random.default_rng(s).shuffle(u)
        for i in range(5):
            te = np.isin(sids, u[i::5]); P[s, te] = HGB(max_depth=3, learning_rate=.05, max_iter=200).fit(E[cols][~te], y[~te]).predict_proba(E[cols][te])[:, 1]
    return P.mean(0)
p0 = oof(REST); p1 = oof(SERVED + REST); pc = oof(["carbs_served"] + REST)
yt, st = y[tgt], sids[tgt]; ppl = np.unique(st); idx = {p: np.where(st == p)[0] for p in ppl}; rng = np.random.default_rng(0)
def boot(a, b, B=2000):
    d = []
    for _ in range(B):
        k = np.concatenate([idx[p] for p in rng.choice(ppl, len(ppl))])
        if yt[k].min() != yt[k].max(): d.append(roc_auc_score(yt[k], a[tgt][k]) - roc_auc_score(yt[k], b[tgt][k]))
    return np.mean(d), np.percentile(d, [2.5, 97.5])
print(f"target cohort n={tgt.sum()}, people={len(ppl)}")
print(f"no meal information (sensor+profile+prior) AUROC {roc_auc_score(yt, p0[tgt]):.3f}")
print(f"+ carbs only                                AUROC {roc_auc_score(yt, pc[tgt]):.3f}")
print(f"+ all served macros                         AUROC {roc_auc_score(yt, p1[tgt]):.3f}")
m, ci = boot(p1, p0); print(f"meal-information gain (served - none): {m:+.3f} 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
CONS = ["carbs", "protein", "fat", "fiber", "kcal"]
p2 = oof(CONS + REST)
print(f"+ consumed macros (retrospective)           AUROC {roc_auc_score(yt, p2[tgt]):.3f}")
full = (E.share.values >= 100)[tgt]
print(f"served macros, fully consumed meals only    AUROC {roc_auc_score(yt[full], p1[tgt][full]):.3f} (n={full.sum()}); no-meal on same meals {roc_auc_score(yt[full], p0[tgt][full]):.3f}")
