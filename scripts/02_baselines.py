"""GBM baselines with nested-CV-style person folds (outer folds only here), reported on the target cohort.
Every fold is fit on its own training participants only; predictions are pooled for reporting."""
import sys, yaml, numpy as np
sys.path.insert(0, "src")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
from glimpse.data import build_meals, add_personal_prior, evaluation_set
from glimpse.evaluation import person_folds, paired_person_bootstrap, load_locked_set

cfg = yaml.safe_load(open("configs/default.yaml"))
locked = set(load_locked_set("splits/locked_final_set.json"))
E = evaluation_set(add_personal_prior(build_meals(cfg["data"]["cgmacros_root"], cfg), cfg))
E = E[~E.sid.isin(locked)].reset_index(drop=True)          # development people only
LAB = ["a1c", "fglu", "fins", "BMI", "Age", "Triglycerides"]
TRAJ = ["slope60", "slope20", "sd120", "min120", "since_meal", "hr_slope", "mets60"]
REST = ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB
SERVED = [f"{m}_served" for m in ["carbs", "protein", "fat", "fiber", "kcal"]]
y, sids = E.spike50.values, E.sid.values
tgt = E.group.isin(cfg["cohort"]["target_groups"]).values

def oof(cols):
    P = np.zeros((len(cfg["cv"]["seeds"]), len(E)))
    for i, s in enumerate(cfg["cv"]["seeds"]):
        for te in person_folds(sids, cfg["cv"]["outer_folds"], s):
            P[i, te] = HGB(max_depth=3, learning_rate=.05, max_iter=200).fit(E[cols][~te], y[~te]).predict_proba(E[cols][te])[:, 1]
    return P.mean(0)

p_eng = oof(SERVED + REST + TRAJ); p_no = oof(SERVED + REST)
print(f"development target cohort: {tgt.sum()} meals, {E[tgt].sid.nunique()} people")
print(f"GBM served + engineered trend AUROC {roc_auc_score(y[tgt], p_eng[tgt]):.3f}")
print(f"GBM served, no trend          AUROC {roc_auc_score(y[tgt], p_no[tgt]):.3f}")
m, lo, hi = paired_person_bootstrap(y[tgt], p_eng[tgt], p_no[tgt], sids[tgt])
print(f"trend features: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
