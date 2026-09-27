"""Fixes for Test 1 (DL vs GBM) and Test 4 (swap monotonicity).
Variants of the from-scratch model, all on the same meals and person folds as the GBM:
  V0 baseline CNN (from test3)
  V1 + hand-crafted trend features in the tabular branch + multi-task (spike + rise) + 5-seed ensemble
  V2 = V1 + self-supervised pretraining of the CNN on unlabelled 4-h CGM windows from TRAINING people only
       (pretext: from the first 2 h, predict the next 2 h of glucose)
  V3 = V2 with a monotone-in-carbs head: logit = h(z) + softplus(a(z)) * carbs  -> swaps can never raise risk
"""
import numpy as np, torch, torch.nn as nn
src = open("/tmp/claude-0/glimpse/test3.py").read().split("res = {")[0]
exec(src)
from sklearn.metrics import roc_auc_score
G = {}  # full Dexcom series per person, for unlabelled pretraining windows
for f in sorted(glob.glob(f"{R}/CGMacros-*/CGMacros-*.csv")):
    sid = int(re.findall(r"(\d+)", os.path.basename(f))[-1])
    d = pd.read_csv(f); d["Timestamp"] = pd.to_datetime(d["Timestamp"]); d = d.set_index("Timestamp").sort_index()
    G[sid] = pd.to_numeric(d["Dexcom GL"], errors="coerce").dropna().groupby(level=0).mean().resample("5min").mean().interpolate(limit=3).values

def ssl_windows(people, n=20000, seed=0):
    r = np.random.default_rng(seed); X, Y = [], []
    people = [p for p in people if p in G and len(G[p]) > 60]
    while len(X) < n:
        g = G[r.choice(people)]; i = r.integers(0, len(g) - 48)
        w = g[i:i + 48]
        if np.isnan(w).any(): continue
        X.append((w[:24] - w[23]) / 20.0); Y.append((w[24:] - w[23]) / 20.0)
    return np.array(X, np.float32), np.array(Y, np.float32)

TAB2 = TAB + TRAJ
class Enc(nn.Module):
    def __init__(s, cin):
        super().__init__(); s.c = nn.Sequential(nn.Conv1d(cin, 16, 5, padding=2), nn.GELU(), nn.Conv1d(16, 16, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(4))
    def forward(s, x): return s.c(x).flatten(1)
class Net2(nn.Module):
    def __init__(s, ntab, mono=False):
        super().__init__(); s.mono = mono
        s.g = Enc(1); s.h = Enc(1); s.tab = nn.Sequential(nn.Linear(ntab, 32), nn.GELU())
        s.body = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + 64 + 32, 32), nn.GELU())
        s.spike = nn.Linear(32, 1); s.rise = nn.Linear(32, 1); s.slope = nn.Linear(32, 1)
    def forward(s, x, t, carbs=None):
        z = s.body(torch.cat([s.g(x[:, :1]), s.h(x[:, 1:]), s.tab(t)], 1))
        logit = s.spike(z).squeeze(1)
        if s.mono: logit = logit + nn.functional.softplus(s.slope(z)).squeeze(1) * carbs
        return logit, s.rise(z).squeeze(1)

CI = TAB2.index("carbs"); KI = TAB2.index("kcal")
def prep_tab(tr):
    X = E[TAB2].astype(float); mu, sd = X[tr].mean(), X[tr].std() + 1e-6
    return mu, sd, np.nan_to_num(((X - mu) / sd).values).astype(np.float32)

def train(tr, te, seed, pre=None, mono=False, Tover=None):
    torch.manual_seed(seed); mu, sd, T = prep_tab(tr)
    if Tover is not None: T = Tover(mu, sd)
    tab = T.copy()
    carbs = torch.tensor(T[:, CI])
    if mono: tab[:, CI] = 0; tab[:, KI] = 0      # carbs/kcal enter ONLY through the monotone path
    u = np.array(sorted(set(sids[tr]))); np.random.default_rng(seed).shuffle(u); va = tr & np.isin(sids, u[:max(3, len(u)//6)]); fi = tr & ~va
    m = Net2(T.shape[1], mono)
    if pre is not None: m.g.load_state_dict(pre)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-3)
    xs, ts, ys = torch.tensor(S), torch.tensor(tab), torch.tensor(y, dtype=torch.float32)
    rv = ((E.rise - 45) / 30).values; rmask = torch.tensor(~np.isnan(rv), dtype=torch.float32); rs = torch.tensor(np.nan_to_num(rv), dtype=torch.float32)
    best, bs, bad = -1, None, 0
    for ep in range(150):
        m.train(); perm = np.random.default_rng(ep + 100 * seed).permutation(np.where(fi)[0])
        for b in range(0, len(perm), 64):
            i = perm[b:b + 64]; lo, ri = m(xs[i] + 0.05 * torch.randn_like(xs[i]), ts[i], carbs[i])
            loss = nn.functional.binary_cross_entropy_with_logits(lo, ys[i]) + 0.3 * (nn.functional.smooth_l1_loss(ri, rs[i], reduction="none") * rmask[i]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.sigmoid(m(xs[va], ts[va], carbs[va])[0]).numpy()
        a = roc_auc_score(y[va], pv)
        if a > best: best, bs, bad = a, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad > 15: break
    m.load_state_dict(bs); m.eval()
    return m, ts, carbs, T

def pretrain(people, seed):
    torch.manual_seed(seed); X, Y = ssl_windows(people, 20000, seed)
    enc = Enc(1); head = nn.Linear(64, 24); opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), 3e-3)
    X, Y = torch.tensor(X)[:, None], torch.tensor(Y)
    for ep in range(6):
        perm = torch.randperm(len(X))
        for b in range(0, len(X), 256):
            i = perm[b:b + 256]; loss = nn.functional.mse_loss(head(enc(X[i])), Y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return enc.state_dict()

out = {k: np.zeros((2, len(E))) for k in ["B4 GBM", "V1 CNN+trend+multitask x5", "V2 = V1 + self-supervised CGM pretraining", "V3 = V2 + monotone-carbs head", "V2 + B4 ensemble", "V3 + B4 ensemble"]}
swap = []
for si, seed in enumerate([0, 1]):
    for te in folds(seed):
        tr = ~te
        out["B4 GBM"][si, te] = gbm(tr, te, TAB2)[1]
        pre = pretrain(sorted(set(sids[tr])), seed)
        for key, kw in [("V1 CNN+trend+multitask x5", {}), ("V2 = V1 + self-supervised CGM pretraining", {"pre": pre}), ("V3 = V2 + monotone-carbs head", {"pre": pre, "mono": True})]:
            ps = []
            for k in range(5):
                m, ts, carbs, T = train(tr, te, 10 * seed + k, **kw)
                with torch.no_grad(): ps.append(torch.sigmoid(m(torch.tensor(S)[te], ts[te], carbs[te])[0]).numpy())
                if key.startswith("V3") and k == 0:  # swap test on held-out meals: halve carbs
                    mu, sd, _ = prep_tab(tr)
                    c2 = torch.tensor(((E.carbs.values[te] / 2 - mu["carbs"]) / sd["carbs"]).astype(np.float32))
                    with torch.no_grad(): p2 = torch.sigmoid(m(torch.tensor(S)[te], ts[te], c2)[0]).numpy()
                    swap.append(p2 - ps[-1])
            out[key][si, te] = np.mean(ps, 0)
    out["V2 + B4 ensemble"][si] = (out["V2 = V1 + self-supervised CGM pretraining"][si] + out["B4 GBM"][si]) / 2
    out["V3 + B4 ensemble"][si] = (out["V3 = V2 + monotone-carbs head"][si] + out["B4 GBM"][si]) / 2
print("\nAUROC (spike >= 50 mg/dL), people held out, mean of 2 fold-seeds")
for k, v in out.items():
    a = [roc_auc_score(y, v[i]) for i in range(2)]
    print(f"  {k:45s} {np.mean(a):.3f}  ({', '.join(f'{x:.3f}' for x in a)})")
d = np.concatenate(swap)
print(f"\nSwap test, monotone DL head (held-out meals, halve carbs): risk lower {np.mean(d < -1e-6):.0%}, unchanged {np.mean(np.abs(d) <= 1e-6):.0%}, higher {np.mean(d > 1e-6):.0%}; median drop {np.median(-d)*100:.1f} pp")
