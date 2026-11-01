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


# ---------- recalibration (fitted on validation only) ----------
def fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling: logistic regression of the outcome on logit(p), nearly unregularised.
    Returns (slope, intercept); slope 1 and intercept 0 leave p unchanged."""
    from sklearn.linear_model import LogisticRegression

    z = _logit(p).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def apply_platt(p: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Recalibrated probability sigmoid(slope * logit(p) + intercept)."""
    return 1 / (1 + np.exp(-(slope * _logit(p) + intercept)))


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))
