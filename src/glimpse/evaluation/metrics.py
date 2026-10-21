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
    if lo > -margin:
        return "non-inferior"
    if hi < -margin:
        return "inferior"
    return "inconclusive"


# ---------- calibration ----------
def ece_equal_mass(p, y, bins=15):
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
    c = np.array(
        [costs["low_risk"], costs["caution"], costs["abstain"]], float
    )  # rows: action; cols: (spike, no spike)
    return float(np.mean(c[actions, np.where(y == 1, 0, 1)]))


def usefulness_report(actions, y, costs, targets):
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
    inside = (y >= lo) & (y <= hi)
    per_person = [inside[sids == p].mean() for p in np.unique(sids)]
    return dict(
        pooled=float(inside.mean()), person_averaged=float(np.mean(per_person)), mean_width=float(np.mean(hi - lo))
    )


# ---------- added for the intermediate evaluation ----------
def ece_equal_width(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """ECE with equal-width probability bins; also returns per-bin count, mean confidence and observed rate."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        conf = float(p[m].mean()) if n else float("nan")
        rate = float(y[m].mean()) if n else float("nan")
        if n:
            ece += abs(conf - rate) * n / len(p)
        table.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                      "mean_predicted": conf, "observed_rate": rate})  # fmt: skip
    return float(ece), table


def person_bootstrap(stat, arrays: tuple[np.ndarray, ...], sids: np.ndarray, B: int = 2000, seed: int = 0):
    """95% percentile CI of `stat(*arrays)` resampling PEOPLE with replacement (meals within a person are dependent).
    Resamples where the statistic is undefined (e.g. one class only) are skipped."""
    rng = np.random.default_rng(seed)
    ppl = np.unique(sids)
    idx = {p: np.where(sids == p)[0] for p in ppl}
    vals = []
    for _ in range(B):
        pick = np.concatenate([idx[p] for p in rng.choice(ppl, len(ppl))])
        try:
            v = stat(*(a[pick] for a in arrays))
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def select_band(p: np.ndarray, y: np.ndarray, costs, max_abstain: float, max_missed_among_cleared: float = 1.0,
                step: float = 0.01) -> tuple[float, float]:  # fmt: skip
    """Low-risk / caution thresholds minimising expected cost, subject to (i) at most `max_missed_among_cleared`
    spikes among meals cleared as low risk and (ii) at most `max_abstain` abstentions. With calibrated risks and
    these costs abstaining is never cheapest on its own, so the band comes from the safety constraint: meals not
    safe enough to clear but below the caution threshold are abstained on. Ties go to fewer abstentions."""
    grid = np.round(np.arange(0, 1 + step / 2, step), 4)
    best = (np.inf, 1.0, 1.0, 0.0, 0.0)
    for lo in grid:
        cleared = p < lo
        if cleared.any() and y[cleared].mean() > max_missed_among_cleared:
            continue
        for hi in grid[grid >= lo]:
            a = decide(p, lo, hi)
            share = float((a == 2).mean())
            if share > max_abstain:
                continue
            key = (expected_cost(a, y, costs), share, hi - lo)
            if key < best[:3]:
                best = (*key, lo, hi)
    return float(best[3]), float(best[4])
