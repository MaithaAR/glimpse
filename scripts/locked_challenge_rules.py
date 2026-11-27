# ruff: noqa
# fmt: off
"""Post hoc, development only: is there a decision rule robust to cohort spike rate?
Reads committed dev answers; writes nothing to the repo. Parameters chosen leave-one-fold-out."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd

SRC = "outputs/evaluation_v2/dev_answers.csv"
df = pd.read_csv(SRC).sort_values(["person_id", "row"]).reset_index(drop=True)
people = np.sort(df.person_id.unique()); pidx = {p: i for i, p in enumerate(people)}
P = df.person_id.map(pidx).to_numpy(); y = df.spike.to_numpy().astype(int)
risk = df.risk.to_numpy(); scored = ~np.isnan(risk)
fold_of_person = df.groupby("person_id").fold.first().reindex(people).to_numpy()
BASE = 0.33  # development spike rate, used only as the shrinkage prior

# causal per-person features: uses only that person's earlier meals
prior_n = np.zeros(len(df)); prior_sp = np.zeros(len(df)); below = np.full(len(df), np.nan); nprior_sc = np.zeros(len(df))
for p in range(len(people)):
    idx = np.where(P == p)[0]; past = []
    for j, i in enumerate(idx):
        prior_n[i] = j; prior_sp[i] = y[idx[:j]].sum(); nprior_sc[i] = len(past)
        if scored[i]:
            if past: below[i] = np.mean(np.array(past) < risk[i])
            past.append(risk[i])
run_rate = (prior_sp + 5 * BASE) / (prior_n + 5)
lg = lambda p: np.log(p / (1 - p))

def actions(kind, par):
    """0 low risk, 1 caution, 2 can't judge (only meals with no risk; no borderline band)."""
    a = np.ones(len(df), int)
    if kind == "served":
        return df.decision.map({"low risk": 0, "caution": 1, "can't judge": 2}).to_numpy()
    if kind == "G":   # one global cut-off
        clear = risk < par[0]
    elif kind == "Q":  # clear the lowest q share of each person's own meals so far (cut-off 0.21 until 3 meals)
        clear = np.where(nprior_sc >= 3, below < par[0], risk < 0.21)
    elif kind == "S":  # cut-off moved by the person's running spike rate, strength g (g<0 stricter for high-rate people)
        clear = risk < par[0] * np.exp(par[1] * (lg(run_rate) - lg(BASE)))
    elif kind == "H":  # global cut-off AND within the person's lowest q share
        clear = (risk < par[0]) & np.where(nprior_sc >= 3, below < par[1], True)
    a[np.nan_to_num(clear, nan=0).astype(bool) & scored] = 0
    a[~scored] = 2
    return a

def pstats(a):
    """per-person sufficient statistics: n, spikes, cleared, cleared spikes, abstain, abstain spikes"""
    cols = [np.ones_like(y), y, a == 0, (a == 0) & (y == 1), a == 2, (a == 2) & (y == 1)]
    return np.stack([np.bincount(P, c.astype(float), len(people)) for c in cols], 1)

def score(S):
    """S: (..., 6) summed stats -> cost reduction, cleared, missed, abstain, targets met"""
    n, sp, cl, cls, ab, abs_ = np.moveaxis(S, -1, 0)
    cost = 5 * cls + ((n - sp) - (cl - cls) - (ab - abs_)) + 2 * abs_ + 0.5 * (ab - abs_)
    cr = 1 - cost / np.minimum(5 * sp, n - sp)
    with np.errstate(invalid="ignore", divide="ignore"):
        missed = np.where(cl > 0, cls / cl, np.nan)
    met = (cr >= .10).astype(int) + (cl / n >= .20) + ((cl > 0) & (missed <= .10)) + (ab / n <= .20)
    return dict(cr=cr, cleared=cl / n, missed=missed, abstain=ab / n, met=met, spike_rate=sp / n)

def draws(members, k, n, seed):
    rng = np.random.default_rng(seed)
    M = np.zeros((n, len(people)))
    for i in range(n): M[i, rng.choice(members, k, replace=False)] = 1
    return M

# same draws as scripts/locked_challenge.py (rng.choice over sorted person ids, seed 20260927)
M_all = draws(np.arange(len(people)), 8, 5000, 20260927)

def summarise(st):
    r = score(M_all @ st); hard = r["spike_rate"] >= 0.40
    f = lambda m: dict(draws=int(m.sum()), at_most_1=float((r["met"][m] <= 1).mean()), mean_met=float(r["met"][m].mean()),
                       missed_median=float(np.nanmedian(r["missed"][m])), missed_over_10pct=float(np.nanmean(r["missed"][m] > .10)),
                       cr_median=float(np.median(r["cr"][m])), cleared_median=float(np.median(r["cleared"][m])))
    pooled = {k: float(v) for k, v in score(st.sum(0)).items()}
    return dict(pooled=pooled, all=f(np.ones(len(hard), bool)), hard=f(hard))

tg = np.round(np.arange(0.10, 0.401, 0.02), 2); qg = np.round(np.arange(0.10, 0.501, 0.05), 2)
GRID = {"G": [(t,) for t in tg], "Q": [(q,) for q in qg],
        "S": list(itertools.product(tg, [-1.0, -0.5, 0.5, 1.0])), "H": list(itertools.product(tg, qg))}
STATS = {k: [pstats(actions(k, p)) for p in g] for k, g in GRID.items()}

out = {"served": summarise(pstats(actions("served", None)))}
for kind, grid in GRID.items():
    st_lofo = np.zeros((len(people), 6)); chosen = {}
    for f in np.unique(fold_of_person):
        train = np.where(fold_of_person != f)[0]
        Mtr = draws(train, 8, 1000, 1000 + int(f))  # criterion: mean targets met over 8-person draws of TRAINING people
        crit = []
        for st in STATS[kind]:
            r = score(Mtr @ st); crit.append((r["met"].mean(), -(r["met"] <= 1).mean(), np.median(r["cr"])))
        b = max(range(len(grid)), key=lambda i: crit[i]); chosen[int(f)] = grid[b]
        st_lofo[fold_of_person == f] = STATS[kind][b][fold_of_person == f]
    res = summarise(st_lofo); res["chosen_per_fold"] = {k: [float(x) for x in v] for k, v in chosen.items()}
    # upper bound, NOT honest: best single parameter picked on the evaluation draws themselves (hard draws)
    orc = [summarise(st) for st in STATS[kind]]
    b = max(range(len(grid)), key=lambda i: orc[i]["hard"]["mean_met"])
    res["oracle_in_sample_hard"] = dict(param=[float(x) for x in grid[b]], **orc[b]["hard"], all_at_most_1=orc[b]["all"]["at_most_1"], all_mean_met=orc[b]["all"]["mean_met"])
    out[kind] = res

# why: spike rate among each person's lowest-risk meals, by the person's own spike rate
hi = np.isin(P, np.where(pstats(actions("served", None))[:, 1] / pstats(actions("served", None))[:, 0] >= .40)[0])
m = scored & hi & (nprior_sc >= 3) & (below < 0.25)
out["diagnostic"] = dict(people_rate_ge40=int(len(np.unique(P[hi]))), lowest_quarter_meals=int(m.sum()), spike_rate_in_them=float(y[m].mean()),
                         break_even_missed=1 / 6, spike_rate_risk_below_0p21_high_people=float(y[scored & hi & (risk < .21)].mean()),
                         n_risk_below_0p21_high_people=int((scored & hi & (risk < .21)).sum()))
Path("outputs/locked_challenge/rules.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    if k == "diagnostic": print(k, v); continue
    print(f"{k:7s} pooled met {v['pooled']['met']:.0f} cr {v['pooled']['cr']:+.3f} clr {v['pooled']['cleared']:.3f} miss {v['pooled']['missed']:.3f} | "
          f"ALL <=1: {v['all']['at_most_1']:.3f} met {v['all']['mean_met']:.2f} miss_med {v['all']['missed_median']:.3f} | "
          f"HARD(n={v['hard']['draws']}) <=1: {v['hard']['at_most_1']:.3f} met {v['hard']['mean_met']:.2f} miss_med {v['hard']['missed_median']:.3f} cr_med {v['hard']['cr_median']:+.3f} clr_med {v['hard']['cleared_median']:.3f}")
    if "chosen_per_fold" in v: print("   chosen", v["chosen_per_fold"], "\n   oracle(hard, in-sample)", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v["oracle_in_sample_hard"].items()})
