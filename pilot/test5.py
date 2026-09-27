"""Pilot under FINAL proposal rules (72-h onboarding, window/gap/share rules, served macros, target-cohort reporting).
A  GBM: served macros + pre-meal glucose + engineered trend + HR + labs + prior           (fair-GBM analogue, logged served macros)
B  NN learned-only: raw 2-h CGM/HR sequence + tabular, NO engineered trend features       (RQ2 learned)
C  NN engineered-only: same net, sequence zeroed, WITH engineered trend features          (RQ2 engineered)
D  NN full: sequence + engineered trend + monotone carbohydrate head                      (fusion analogue)
D+A ensemble. Folds split all 45 people; AUROC reported on target-cohort meals (healthy + prediabetes)."""
import numpy as np, pandas as pd, torch, torch.nn as nn, glob, re, os, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
exec(open("analysis_final_rules.py").read().split('MAC = ["carbs"')[0].replace("print(", "(lambda *a, **k: None)("))
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
E = E.reset_index(drop=True)
seq = {}
for f in sorted(glob.glob(f"{R}/CGMacros-*/CGMacros-*.csv")):
    sid = int(re.findall(r"(\d+)", os.path.basename(f))[-1]); d = pd.read_csv(f); d["Timestamp"] = pd.to_datetime(d["Timestamp"]); d = d.set_index("Timestamp").sort_index()
    g = pd.to_numeric(d["Dexcom GL"], errors="coerce").dropna().groupby(level=0).mean().resample("5min").mean()
    hr = pd.to_numeric(d.get("HR", pd.Series(np.nan, index=d.index)), errors="coerce").resample("5min").mean()
    for t in E[E.sid == sid].t:
        idx = pd.date_range(t - pd.Timedelta("115min"), t, freq="5min")
        seq[(sid, t)] = np.stack([g.reindex(idx, method="nearest", tolerance=pd.Timedelta("3min")).interpolate(limit_direction="both").values,
                                  hr.reindex(idx, method="nearest", tolerance=pd.Timedelta("3min")).interpolate(limit_direction="both").values])
S = np.stack([seq[(s, t)] for s, t in zip(E.sid, E.t)]).astype(np.float32)
S[:, 0] = (S[:, 0] - E.base.values[:, None]) / 20.0; S[:, 1] = (S[:, 1] - np.nanmean(S[:, 1])) / (np.nanstd(S[:, 1]) + 1e-6); S = np.nan_to_num(S)
MACS = ["carbs_served", "protein_served", "fat_served", "fiber_served", "kcal_served"]
LAB = ["a1c", "fglu", "fins", "BMI", "Age", "Triglycerides"]; TRAJ = ["slope60", "slope20", "sd120", "min120", "since_meal", "hr_slope", "mets60"]
TAB = MACS + ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB
y = E.s50.values; sids = E.sid.values; tgt = E.group.isin(["healthy", "prediabetes"]).values
print(f"meals {len(E)} ({tgt.sum()} target), people {E.sid.nunique()}")
def folds(seed, k=5):
    u = np.array(sorted(set(sids))); r = np.random.default_rng(seed); r.shuffle(u); return [np.isin(sids, u[i::k]) for i in range(k)]
class Enc(nn.Module):
    def __init__(s): super().__init__(); s.c = nn.Sequential(nn.Conv1d(1, 8, 5, padding=2), nn.GELU(), nn.Conv1d(8, 8, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(4))
    def forward(s, x): return s.c(x).flatten(1)
class Net(nn.Module):
    def __init__(s, ntab, mono):
        super().__init__(); s.mono = mono; s.g = Enc(); s.h = Enc(); s.tab = nn.Sequential(nn.Linear(ntab, 32), nn.GELU())
        s.body = nn.Sequential(nn.Dropout(0.3), nn.Linear(32 + 32 + 32, 32), nn.GELU()); s.spike = nn.Linear(32, 1); s.rise = nn.Linear(32, 1); s.slope = nn.Linear(32, 1)
    def forward(s, x, t, c):
        z = s.body(torch.cat([s.g(x[:, :1]), s.h(x[:, 1:]), s.tab(t)], 1)); lo = s.spike(z).squeeze(1)
        if s.mono: lo = lo + nn.functional.softplus(s.slope(z)).squeeze(1) * c
        return lo, s.rise(z).squeeze(1)
def fit(tr, te, seed, cols, use_seq, mono):
    torch.manual_seed(seed); X = E[cols].astype(float); mu, sd = X[tr].mean(), X[tr].std() + 1e-6
    T = np.nan_to_num(((X - mu) / sd).values).astype(np.float32); c = torch.tensor(T[:, cols.index("carbs_served")])
    if mono: T[:, cols.index("carbs_served")] = 0; T[:, cols.index("kcal_served")] = 0
    xs = torch.tensor(S if use_seq else np.zeros_like(S)); ts = torch.tensor(T); ys = torch.tensor(y, dtype=torch.float32)
    rv = ((E.rise - 45) / 30).values; rm = torch.tensor(~np.isnan(rv), dtype=torch.float32); rs = torch.tensor(np.nan_to_num(rv), dtype=torch.float32)
    u = np.array(sorted(set(sids[tr]))); np.random.default_rng(seed).shuffle(u); va = tr & np.isin(sids, u[:max(3, len(u) // 6)]); fi = tr & ~va
    m = Net(T.shape[1], mono); opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-3); best, bs, bad = -1, None, 0
    for ep in range(120):
        m.train(); perm = np.random.default_rng(ep + 100 * seed).permutation(np.where(fi)[0])
        for b in range(0, len(perm), 64):
            i = perm[b:b + 64]; lo, ri = m(xs[i] + 0.05 * torch.randn_like(xs[i]) * float(use_seq), ts[i], c[i])
            loss = nn.functional.binary_cross_entropy_with_logits(lo, ys[i]) + 0.3 * (nn.functional.smooth_l1_loss(ri, rs[i], reduction="none") * rm[i]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): a = roc_auc_score(y[va], m(xs[va], ts[va], c[va])[0].numpy())
        if a > best: best, bs, bad = a, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad > 12: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): return torch.sigmoid(m(xs[te], ts[te], c[te])[0]).numpy()
names = ["A GBM (logged served macros, engineered trend)", "B NN learned sequence, no engineered trend", "C NN engineered trend, no sequence", "D NN sequence + trend + monotone carbs", "D + A ensemble"]
out = {k: np.zeros((2, len(E))) for k in names}
for si, seed in enumerate([0, 1]):
    for te in folds(seed):
        tr = ~te
        out[names[0]][si, te] = HGB(max_depth=3, learning_rate=.05, max_iter=200).fit(E[TAB + TRAJ][tr], y[tr]).predict_proba(E[TAB + TRAJ][te])[:, 1]
        for nm, cols, us, mo in [(names[1], TAB, True, False), (names[2], TAB + TRAJ, False, False), (names[3], TAB + TRAJ, True, True)]:
            out[nm][si, te] = np.mean([fit(tr, te, 10 * seed + k, cols, us, mo) for k in range(3)], 0)
    out[names[4]][si] = (out[names[3]][si] + out[names[0]][si]) / 2
    print(f"fold-seed {seed} done", flush=True)
P = {k: v.mean(0) for k, v in out.items()}
yt, st = y[tgt], sids[tgt]
print("\nAUROC on TARGET-cohort meals (people held out; mean of 2 fold-seeds, 3 net seeds each)")
for k in names: print(f"  {k:48s} {roc_auc_score(yt, P[k][tgt]):.3f}")
rng = np.random.default_rng(0); ppl = np.unique(st); idx = {p: np.where(st == p)[0] for p in ppl}
def paired(a, b, B=2000):
    d = []
    for _ in range(B):
        pick = np.concatenate([idx[p] for p in rng.choice(ppl, len(ppl))])
        if yt[pick].min() != yt[pick].max(): d.append(roc_auc_score(yt[pick], P[a][tgt][pick]) - roc_auc_score(yt[pick], P[b][tgt][pick]))
    return np.mean(d), np.percentile(d, [2.5, 97.5])
for a, b, lab in [(names[1], names[2], "RQ2 pilot: learned (B) - engineered (C)"), (names[3], names[0], "fusion D - GBM A"), (names[4], names[0], "ensemble - GBM A")]:
    m, ci = paired(a, b); print(f"  {lab:42s} diff {m:+.3f}  95% person-bootstrap CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
