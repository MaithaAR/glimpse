"""Evaluation utilities matching the proposal's prespecified analyses."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------- splits ----------
def load_locked_set(path: str) -> list[int]:
    """Load the frozen 8-person locked set and verify its SHA-256 hash."""
    d = json.load(open(path))
    payload = {k: d[k] for k in d if k != "sha256"}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == d["sha256"], (
        "locked set modified!"
    )
    return d["healthy"] + d["prediabetes"]


def person_folds(sids: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Boolean test masks; every learned component of an outer fold must be fit on ~mask only."""
    u = np.array(sorted(set(sids)))
    np.random.default_rng(seed).shuffle(u)
    return [np.isin(sids, u[i::k]) for i in range(k)]


# ---------- discrimination ----------
def paired_person_bootstrap(y, pa, pb, sids, B=2000, seed=0):
    """Mean and 95% CI of AUROC(pa) - AUROC(pb), resampling PEOPLE (meals within a person are dependent)."""
    rng = np.random.default_rng(seed)
    ppl = np.unique(sids)
    idx = {p: np.where(sids == p)[0] for p in ppl}
    d = []
    for _ in range(B):
        pick = np.concatenate([idx[p] for p in rng.choice(ppl, len(ppl))])
        if y[pick].min() != y[pick].max():
            d.append(roc_auc_score(y[pick], pa[pick]) - roc_auc_score(y[pick], pb[pick]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


def noninferiority(lo: float, hi: float, margin: float = 0.03) -> str:
    """Classify a 95% interval of an AUROC difference against the prespecified margin."""
    if lo > -margin:
        return "non-inferior"
    if hi < -margin:
        return "inferior"
    return "inconclusive"


# ---------- calibration ----------
def ece_equal_mass(p, y, bins=15):
    """Expected calibration error with bins holding equal numbers of predictions."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(abs(p[c].mean() - y[c].mean()) * len(c) / len(p) for c in chunks if len(c)))


# ---------- decisions ----------
ACTIONS = ("low_risk", "caution", "abstain")


def decide(p, lo_thr, hi_thr, abstain_mask=None):
    """low risk if p < lo_thr, caution if p >= hi_thr, abstain in between or where abstain_mask is True."""
    a = np.where(p < lo_thr, 0, np.where(p >= hi_thr, 1, 2))
    if abstain_mask is not None:
        a = np.where(abstain_mask, 2, a)
    return a


def expected_cost(actions, y, costs):
    """Mean cost of the chosen actions (0 low risk, 1 caution, 2 abstain) given the true spike labels."""
    c = np.array(
        [costs["low_risk"], costs["caution"], costs["abstain"]], float
    )  # rows: action; cols: (spike, no spike)
    return float(np.mean(c[actions, np.where(y == 1, 0, 1)]))


def usefulness_report(actions, y, costs, targets):
    """Share of each action, cost vs the best trivial policy, spikes among cleared meals, and the verdict."""
    trivial = min(expected_cost(np.zeros_like(y), y, costs), expected_cost(np.ones_like(y), y, costs))
    cost = expected_cost(actions, y, costs)
    cleared = actions == 0
    missed = float(y[cleared].mean()) if cleared.any() else float("nan")
    rep = dict(
        share_low_risk=float(cleared.mean()),
        share_caution=float((actions == 1).mean()),
        share_abstain=float((actions == 2).mean()),
        cost=cost,
        best_trivial_cost=trivial,
        cost_reduction=1 - cost / trivial if trivial > 0 else float("nan"),
        missed_among_cleared=missed,
        n_cleared=int(cleared.sum()),
    )
    u = targets["usefulness"]
    rep["useful"] = bool(
        cleared.any()
        and rep["cost_reduction"] >= u["min_cost_reduction"]
        and rep["share_low_risk"] >= u["min_cleared"]
        and missed <= u["max_missed_among_cleared"]
        and rep["share_abstain"] <= u["max_abstain"]
    )
    return rep


def choose_thresholds(p, y, costs, grid=None):
    """(lo, hi) with lo <= hi that give the lowest mean cost: low risk below lo, caution from hi, abstain between.
    Ties go to the first pair found (smallest lo, then smallest hi), so the choice is deterministic."""
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2) if grid is None else grid
    best = (float("inf"), 0.0, 0.0)
    for lo in grid:
        for hi in grid[grid >= lo]:
            c = expected_cost(decide(p, lo, hi), y, costs)
            if c < best[0] - 1e-12:
                best = (c, float(lo), float(hi))
    return best[1], best[2], best[0]


def cost_only_thresholds(costs):
    """Cut-offs that follow from the costs alone, if the probability p is honest (calibrated): each answer's
    expected cost is a straight line in p; pick the cheapest. Low risk a*p, caution b*(1-p), abstain c*p + d*(1-p)
    with (a, 0), (0, b), (c, d) = low_risk, caution, abstain. Uses no data."""
    a, b = costs["low_risk"][0], costs["caution"][1]
    c, d = costs["abstain"]
    den_lo, den_hi = a - c + d, b - d + c
    if den_lo > 0 and den_hi > 0:
        lo, hi = d / den_lo, (b - d) / den_hi
        if 0 <= lo < hi <= 1:
            return float(lo), float(hi)
    # abstaining is never the cheapest answer: one cut-off where low risk and caution cost the same
    return float(b / (a + b)), float(b / (a + b))


def person_bootstrap_ci(sids, fn, B=2000, seed=0):
    """Point value and 95% range of fn(meal indices), resampling PEOPLE with replacement. fn may return NaN for a
    draw where it is undefined (e.g. only one class); those draws are skipped."""
    rng = np.random.default_rng(seed)
    ppl = np.unique(sids)
    idx = {p: np.where(sids == p)[0] for p in ppl}
    vals = []
    for _ in range(B):
        v = fn(np.concatenate([idx[p] for p in rng.choice(ppl, len(ppl))]))
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(fn(np.arange(len(sids)))), float(lo), float(hi)


def reliability(p, y, bins=15):
    """Equal-width bins: mean predicted, observed rate and count per non-empty bin, and the ECE
    (count-weighted mean gap)."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = [(float(p[b == k].mean()), float(y[b == k].mean()), int((b == k).sum()))
            for k in range(bins) if (b == k).any()]  # fmt: skip
    ece = sum(n * abs(m - o) for m, o, n in rows) / len(p)
    return rows, float(ece)


# ---------- conformal rise intervals ----------
def participant_weighted_cqr_quantile(lo_pred, hi_pred, y, sids, alpha=0.2):
    """Conformalised quantile regression score with each calibration person weighted 1/(their meal count).
    No coverage guarantee is claimed under within-person dependence; coverage is evaluated empirically."""
    s = np.maximum(lo_pred - y, y - hi_pred)
    w = np.array([1.0 / np.sum(sids == p) for p in sids])
    o = np.argsort(s)
    cw = np.cumsum(w[o]) / w.sum()
    return float(s[o][np.searchsorted(cw, 1 - alpha)])


def coverage_report(lo, hi, y, sids):
    """Pooled and person-averaged coverage of predicted rise intervals, and their mean width."""
    inside = (y >= lo) & (y <= hi)
    per_person = [inside[sids == p].mean() for p in np.unique(sids)]
    return dict(
        pooled=float(inside.mean()), person_averaged=float(np.mean(per_person)), mean_width=float(np.mean(hi - lo))
    )


def calibration_by_person(person: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict:
    """Mean predicted risk vs observed spike rate per person, and for the people below/above the median spike rate.

    A strong negative correlation between a person's spike rate and (predicted - observed) means the model cannot
    tell rare from frequent spikers: it over-predicts for the first and under-predicts for the second.
    """
    ids = np.unique(person)
    n = np.array([(person == q).sum() for q in ids])
    obs = np.array([y[person == q].mean() for q in ids])
    pred = np.array([p[person == q].mean() for q in ids])
    low = obs < np.median(obs)

    def half(m: np.ndarray) -> dict:
        return {"people": int(m.sum()), "predicted": float(np.average(pred[m], weights=n[m])),
                "observed": float(np.average(obs[m], weights=n[m]))}  # fmt: skip

    return {"corr_spike_rate_vs_gap": float(np.corrcoef(obs, pred - obs)[0, 1]),
            "rare_spikers": half(low), "frequent_spikers": half(~low)}  # fmt: skip
