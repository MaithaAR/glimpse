"""Extra confidence tests for GLIMPSE (person-level CV throughout).
T1: small from-scratch sequence network vs GBM on the same meals/folds
T2: calibration (ECE) raw vs cross-fitted Platt scaling
T3: cost / abstention feasibility of the proposal's targets
T4: swap monotonicity (does halving carbs lower predicted risk?)"""
import numpy as np, pandas as pd, torch, torch.nn as nn, warnings, re, glob, os
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
exec(open("/tmp/claude-0/glimpse/analysis2.py").read().split("MAC = [")[0].replace("print(", "(lambda *a, **k: None)("))

# --- add raw pre-meal sequences (24 x 5-min for glucose, HR resampled) -------------------------
seq = {}
for f in sorted(glob.glob(f"{R}/CGMacros-*/CGMacros-*.csv")):
    sid = int(re.findall(r"(\d+)", os.path.basename(f))[-1])
    d = pd.read_csv(f); d["Timestamp"] = pd.to_datetime(d["Timestamp"]); d = d.set_index("Timestamp").sort_index()
    g = pd.to_numeric(d["Dexcom GL"], errors="coerce").dropna().groupby(level=0).mean().resample("5min").mean()
    hr = pd.to_numeric(d.get("HR", pd.Series(np.nan, index=d.index)), errors="coerce").resample("5min").mean()
    for t in E[E.sid == sid].t:
        idx = pd.date_range(t - pd.Timedelta("115min"), t, freq="5min")
        gs = g.reindex(idx, method="nearest", tolerance=pd.Timedelta("3min")).interpolate(limit_direction="both").values
        hs = hr.reindex(idx, method="nearest", tolerance=pd.Timedelta("3min")).interpolate(limit_direction="both").values
        seq[(sid, t)] = np.stack([gs, hs])
E = E[[(s, t) in seq for s, t in zip(E.sid, E.t)]].reset_index(drop=True)
S = np.stack([seq[(s, t)] for s, t in zip(E.sid, E.t)]).astype(np.float32)
S[:, 0] = (S[:, 0] - E.base.values[:, None]) / 20.0                       # glucose relative to pre-meal level
S[:, 1] = (S[:, 1] - np.nanmean(S[:, 1])) / (np.nanstd(S[:, 1]) + 1e-6)
S = np.nan_to_num(S)
MAC = ["carbs", "protein", "fat", "fiber", "kcal"]; LAB = ["a1c", "fglu", "fins", "BMI", "Age", "Triglycerides"]
TAB = MAC + ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB
TRAJ = ["slope60", "slope20", "sd120", "min120", "since_meal", "hr_slope", "mets60"]
y = E.s50.values; sids = E.sid.values
print(f"meals {len(E)} from {E.sid.nunique()} people, spike rate {y.mean():.0%}")

def folds(seed, k=5):
    u = np.array(sorted(set(sids))); r = np.random.default_rng(seed); r.shuffle(u)
    return [np.isin(sids, u[i::k]) for i in range(k)]

class Net(nn.Module):  # from-scratch: 1-D CNN over 2h sensor sequence + MLP over tabular -> spike logit
    def __init__(s, ntab):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv1d(2, 16, 5, padding=2), nn.GELU(), nn.Conv1d(16, 16, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(4))
        s.tab = nn.Sequential(nn.Linear(ntab, 32), nn.GELU())
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + 32, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(s, x, t): return s.head(torch.cat([s.cnn(x).flatten(1), s.tab(t)], 1)).squeeze(1)

def fit_net(tr, te, seed):
    torch.manual_seed(seed)
    X = E[TAB].astype(float); mu, sd = X[tr].mean(), X[tr].std() + 1e-6
    T = np.nan_to_num(((X - mu) / sd).values).astype(np.float32)
    # inner person split for early stopping
    u = np.array(sorted(set(sids[tr]))); np.random.default_rng(seed).shuffle(u); va = tr & np.isin(sids, u[:max(3, len(u)//6)]); fi = tr & ~va
    m = Net(T.shape[1]); opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-3); lossf = nn.BCEWithLogitsLoss()
    xs, ts, ys = torch.tensor(S), torch.tensor(T), torch.tensor(y, dtype=torch.float32)
    best, bstate, bad = -1, None, 0
    for ep in range(200):
        m.train(); perm = np.random.default_rng(ep).permutation(np.where(fi)[0])
        for b in range(0, len(perm), 64):
            i = perm[b:b+64]; xb = xs[i] + 0.05 * torch.randn_like(xs[i])   # jitter augmentation
            opt.zero_grad(); lossf(m(xb, ts[i]), ys[i]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.sigmoid(m(xs[va], ts[va])).numpy()
        a = roc_auc_score(y[va], pv) if y[va].min() != y[va].max() else 0
        if a > best: best, bstate, bad = a, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad > 20: break
    m.load_state_dict(bstate); m.eval()
    with torch.no_grad(): return torch.sigmoid(m(xs[te], ts[te])).numpy()

def gbm(tr, te, cols, mono=None):
    m = HGB(max_depth=3, learning_rate=.05, max_iter=200, monotonic_cst=mono); m.fit(E[cols][tr], y[tr]); return m, m.predict_proba(E[cols][te])[:, 1]

res = {k: np.zeros((3, len(E))) for k in ["GBM tabular (B3)", "GBM + hand-crafted trend (B4)", "DL from scratch: CNN on raw 2-h sequence + tabular", "Ensemble DL + B4"]}
for si, seed in enumerate([0, 1, 2]):
    for te in folds(seed):
        tr = ~te
        res["GBM tabular (B3)"][si, te] = gbm(tr, te, TAB)[1]
        res["GBM + hand-crafted trend (B4)"][si, te] = gbm(tr, te, TAB + TRAJ)[1]
        res["DL from scratch: CNN on raw 2-h sequence + tabular"][si, te] = fit_net(tr, te, seed)
    res["Ensemble DL + B4"][si] = (res["GBM + hand-crafted trend (B4)"][si] + res["DL from scratch: CNN on raw 2-h sequence + tabular"][si]) / 2
print("\nT1  AUROC per seed (mean)  — same meals, same person folds")
for k, v in res.items():
    a = [roc_auc_score(y, v[i]) for i in range(3)]
    print(f"  {k:52s} {np.mean(a):.3f}  (seeds {', '.join(f'{x:.3f}' for x in a)})")

# --- T2 calibration ------------------------------------------------------------------------------
def ece(p, yy, bins=10):
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return sum(abs(p[b == i].mean() - yy[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())
p = res["Ensemble DL + B4"].mean(0)
pc = np.zeros_like(p)  # cross-fitted Platt: calibrator for each fold learned on other folds' predictions
for te in folds(7):
    lr = LogisticRegression().fit(np.log(p[~te] / (1 - p[~te]))[:, None], y[~te]); pc[te] = lr.predict_proba(np.log(p[te] / (1 - p[te]))[:, None])[:, 1]
print(f"\nT2  ECE raw {ece(p, y):.3f} -> after cross-fitted Platt scaling {ece(pc, y):.3f}  (target <= 0.05)")

# --- T3 cost / abstention ------------------------------------------------------------------------
print("\nT3  Decision policy on calibrated risk: clear if p<lo, caution if p>=hi, dietitian queue in between")
print("    (cost: missed spike 5, unnecessary caution 1, dietitian 0.5; per meal)")
best = None
for lo in np.arange(.05, .5, .025):
    for hi in np.arange(lo, .9, .025):
        clr, cau = pc < lo, pc >= hi; ab = ~clr & ~cau
        if ab.mean() > .20: continue
        miss = (y[clr] == 1).mean() if clr.any() else 0
        cost = (5 * ((y == 1) & clr).sum() + 1 * ((y == 0) & cau).sum() + .5 * ab.sum()) / len(y)
        if best is None or cost < best[0]: best = (cost, lo, hi, clr.mean(), ab.mean(), miss)
c0 = min((5 * ((y == 1) & (pc < t)).sum() + ((y == 0) & (pc >= t)).sum()) / len(y) for t in np.arange(.05, .95, .01))
print(f"    best: lo={best[1]:.2f} hi={best[2]:.2f} -> cleared {best[3]:.0%}, dietitian {best[4]:.0%}, missed spikes among cleared {best[5]:.0%}, cost/meal {best[0]:.3f}")
print(f"    same model without the dietitian tier: best cost/meal {c0:.3f};  'always caution' cost/meal {(y == 0).mean():.3f}")

# --- T4 swap monotonicity ------------------------------------------------------------------------
tr = np.ones(len(E), bool)
m_free, _ = gbm(tr, tr, TAB + TRAJ)
mono = [1 if c in ("carbs", "kcal") else 0 for c in TAB + TRAJ]
m_mono, _ = gbm(tr, tr, TAB + TRAJ, mono)
Xs = E[TAB + TRAJ].copy(); Xs["kcal"] = Xs.kcal - 2 * Xs.carbs; Xs["carbs"] = Xs.carbs / 2   # swap: half the carbs
for name, m in [("unconstrained GBM", m_free), ("GBM with monotone carbs constraint", m_mono)]:
    d = m.predict_proba(Xs)[:, 1] - m.predict_proba(E[TAB + TRAJ])[:, 1]
    print(f"\nT4  {name:36s} risk lower after halving carbs: {np.mean(d < 0):.0%}, unchanged {np.mean(d == 0):.0%}, higher {np.mean(d > 0):.0%}")
