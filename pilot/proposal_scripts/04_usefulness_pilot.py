"""Pilot check of the proposal's usefulness targets on out-of-fold GBM predictions from 03_photo_experiment.py.

Per model (none | reference | photo): cross-fitted Platt calibration (5 person folds), then the cost-optimal action
under the proposal's cost matrix (low risk 5/0, caution 0/1, abstain 2/0.5 for spike / no spike), scored on isolated
target-cohort meals. Reports cost reduction vs the best trivial policy, share cleared, spike rate among cleared,
abstention and ECE (15 equal-mass bins). Exploratory: isolated meals only, GBM not the final neural fusion.
Usage: python3 scripts/04_usefulness_pilot.py [results/photo_experiment_oof.npz]
"""
import sys, yaml
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "src")
from glimpse.evaluation import ece_equal_mass, usefulness_report, person_folds

cfg = yaml.safe_load(open("configs/default.yaml"))
d = np.load(sys.argv[1] if len(sys.argv) > 1 else "results/photo_experiment_oof.npz")
y, sids, tgt = d["y"], d["sids"], d["target"]
C = cfg["costs"]
M = np.array([C["low_risk"], C["caution"], C["abstain"]], float)  # rows: action; cols: (spike, no spike)
lines = []
for name in ["none", "photo", "reference"]:
    p = np.clip(d[name], 1e-6, 1 - 1e-6); z = np.log(p / (1 - p)); q = np.zeros_like(p)
    for te in person_folds(sids, 5, seed=0):
        q[te] = LogisticRegression().fit(z[~te, None], y[~te]).predict_proba(z[te, None])[:, 1]
    act = np.stack([M[a, 0] * q + M[a, 1] * (1 - q) for a in range(3)]).argmin(0)
    r = usefulness_report(act[tgt], y[tgt], C, cfg["targets"])
    lines.append(f"{name:10s} cost {r['cost']:.3f} vs best trivial {r['best_trivial_cost']:.3f} (reduction {r['cost_reduction']:+.1%}) | "
          f"cleared {r['share_low_risk']:.1%} (spiking {r['missed_among_cleared']:.1%}) | abstain {r['share_abstain']:.1%} | "
          f"ECE {ece_equal_mass(q[tgt], y[tgt], cfg['targets']['ece_bins']):.3f} | meets targets: {r['useful']}")
print("\n".join(lines)); open("results/usefulness_pilot.log", "w").write("\n".join(lines) + "\n")
